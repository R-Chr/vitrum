import pandas as pd
import numpy as np
import itertools
import math
from pathlib import Path
from vitrum.glass_Atoms import GlassAtoms
from tqdm import tqdm
from scipy.stats import norm
from ase import Atom, Atoms
from scipy import integrate
from typing import List, Union, Optional, Tuple
import logging
from collections import defaultdict
from ase.neighborlist import neighbor_list

def gaussian_broadening(g_r: np.ndarray, r: np.ndarray, Q_max: float) -> np.ndarray:
    """
    Broaden the RDF using a Gaussian convolution.

    Args:
        g_r (np.ndarray): The RDF values.
        r (np.ndarray): The r values.
        Q_max (float): The maximum Q value.

    Returns:
        np.ndarray: Broadened RDF.
    """
    delta_r = r[np.newaxis, :] - r[:, np.newaxis]
    sum_r = r[np.newaxis, :] + r[:, np.newaxis]
    FWHM = 5.437 / Q_max
    sigma = FWHM / 2.355
    foubroad = g_r * (norm.pdf(delta_r, 0, sigma) - norm.pdf(sum_r, 0, sigma))
    dist_broad = np.trapz(foubroad, r)
    return dist_broad


class Scattering:
    """
    Class for calculating scattering functions from glass structures.
    """
    def __init__(
        self,
        atoms: Union[List[Atoms], Atoms],
        qmin: float = 0.5,
        qmax: float = 20.0,
        rrange: Optional[float] = None,
        nbin: int = 500,
        neutron_scattering_coef: Optional[List[float]] = None,
        x_ray_scattering_coef: Optional[np.ndarray] = None,
        disable_progress: bool = False,
        use_neighborhood: bool = False
    ):
        """
        Initializes a new instance of the class with the given atoms.

        Args:
            atoms (Union[List[Atoms], Atoms]): A list of Atoms objects or a single Atoms object.
            qmin (float, optional): The minimum q-value to use. Defaults to 0.5.
            qmax (float, optional): The maximum q-value to use. Defaults to 20.
            rrange (float, optional): The range of r-values to use. If None, defaults to min(cell_dim)/2.
            nbin (int, optional): The number of bins to use. Defaults to 500.
            neutron_scattering_coef (List[float], optional): A list of custom neutron scattering lengths. Defaults to None.
              If None, the default coefficients from Neutron News, Vol. 3, No. 3, 1992, pp. 29-37 are used.
            x_ray_scattering_coef (np.ndarray, optional): A list of custom x-ray scattering coefficients. Defaults to None.
              If None, the default coefficients from International Tables for Crystallography (2006). Vol. C. ch. 6.1,
              pp. 554-595 are used.
            disable_progress (bool, optional): Whether to disable the progress bar. Defaults to False.
        """

        if isinstance(atoms, list):
            atom_list = atoms
        else:
            atom_list = [atoms]
            
        self.atom_list = [GlassAtoms(atom) for atom in atom_list]
        script_dir = Path(__file__).parent

        cell_lengths = np.diag(atom_list[0].get_cell())
        half_min_dim = np.min(cell_lengths) / 2

        if rrange:
            if rrange > half_min_dim:
                logging.warning(
                    f"Specified rrange ({rrange:.2f}) exceeds half the shortest cell length ({half_min_dim:.2f}). "
                    "This may violate the Minimum Image Convention."
                )
            self.rrange = rrange
        else:
            # Default to half the shortest cell dimension
            self.rrange = half_min_dim

        self.nbin = nbin
        edges = np.linspace(0, self.rrange, self.nbin + 1)
        self.xval = (edges[:-1] + edges[1:]) / 2.0
        self.volbin = (4 / 3) * np.pi * (edges[1:]**3 - edges[:-1]**3)

        self.qval = np.linspace(qmin, qmax, self.nbin)
        self.chemical_symbols = atom_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.pairs = [pair for pair in itertools.product(self.species, repeat=2)]
        self.c = [self.chemical_symbols.count(i) / len(self.chemical_symbols) for i in self.species]
        self.volume = atom_list[0].get_volume()
        self.aveden = len(atom_list[0]) / self.volume
        self.atomic_numbers = [Atom(atom).number for atom in self.species]
        self.disable_progress = disable_progress

        # Neutron
        if neutron_scattering_coef is None:
            self.scattering_lengths = pd.read_csv(script_dir / "scattering_lengths.csv", sep=";", decimal=",")
            self.b = np.array(
                [self.scattering_lengths[self.scattering_lengths["Isotope"] == i]["b"] for i in self.species]
            ).flatten()
        else:
            self.b = neutron_scattering_coef

        self.cb = [i * j for i, j in zip(self.c, self.b)]
        self.timesby = [pair[0] * pair[1] for pair in itertools.product(self.cb, repeat=2)]

        # X-ray
        if x_ray_scattering_coef is None:
            x_ray_scattering_coef_df = pd.read_csv(script_dir / "x_ray_scattering_factor_coefficients.csv", sep=",")
            x_ray_scattering_coef_arr = np.array(
                [x_ray_scattering_coef_df[x_ray_scattering_coef_df["Element"] == i] for i in self.species]
            ).reshape([len(self.species), 10])
        else:
            x_ray_scattering_coef_arr = x_ray_scattering_coef

        self.x_ray_a = x_ray_scattering_coef_arr[:, [1, 3, 5, 7]]
        self.x_ray_b = x_ray_scattering_coef_arr[:, [2, 4, 6, 8]]
        self.x_ray_c = x_ray_scattering_coef_arr[:, [9]]

        self.f_i = []

        for ind in range(len(self.species)):

            self.f_i.append(
                np.sum(
                    [
                        self.x_ray_a[ind][i] * np.exp(-self.x_ray_b[ind][i] * ((self.qval) / (4 * np.pi)) ** 2)
                        for i in range(4)
                    ],
                    axis=0,
                )
                + self.x_ray_c[ind]
            )

        self.xray_cb = [i * j for i, j in zip(self.c, self.f_i)]
        self.xray_timesby = [pair[0] * pair[1] for pair in itertools.product(self.xray_cb, repeat=2)]

        self.approx_xray_cb = np.array([i * j for i, j in zip(self.c, self.atomic_numbers)])
        self.approx_xray_timesby = np.array(
            [pair[0] * pair[1] for pair in itertools.product(self.approx_xray_cb, repeat=2)]
        )

        if use_neighborhood:
            self.partial_pdfs = self.calculate_partial_pdfs_neighborhood()
        else:
            self.partial_pdfs = self.calculate_partial_pdfs()

    def calculate_partial_pdfs(self) -> np.ndarray:
        """
        Calculate partial PDFs for all pairs from full distance matrix. 
        Scales as O(N^2) with number of atoms, so may be slow for large systems, can be more efficient when using large cutoffs.
        
        Returns:
            np.ndarray: Array of partial PDFs.
        """
        edges = np.linspace(0, self.rrange, self.nbin + 1)
        volbin = (4 / 3) * np.pi * (edges[1:]**3 - edges[:-1]**3)

        pdf_sum = np.zeros((len(self.pairs), self.nbin))
        n_frames = len(self.atom_list)

        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            distances = atom.get_dist()
            symbols = np.array(atom.get_chemical_symbols())
            volume = atom.get_volume()
            
            for pair_ind, pair in enumerate(self.pairs):
                idx_1 = np.flatnonzero(symbols == pair[0])
                idx_2 = np.flatnonzero(symbols == pair[1])           
                dist_list = distances[np.ix_(idx_1, idx_2)]
                h, _ = np.histogram(dist_list, bins=self.nbin, range=(0, self.rrange))
                if pair[0] == pair[1]:
                    h[0] = 0
                number_density_factor = (len(idx_1) * len(idx_2)) / volume
                current_pdf = (h / volbin) / number_density_factor
                pdf_sum[pair_ind, :] += current_pdf
        return pdf_sum / n_frames


    def calculate_partial_pdfs_neighborhood(self):
        """
        Calculate partial PDFs using O(N) neighbor lists

        Returns:
            np.ndarray: Array of partial PDFs.
        """
        all_frame_data = defaultdict(list)
        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            symbols = np.array(self.chemical_symbols)
            volume = self.volume
            i_list, j_list, d_list = neighbor_list("ijd", a=atom, cutoff=self.rrange)
            pair_distances = defaultdict(list)
            for i_idx, j_idx, d in zip(i_list, j_list, d_list):
                pair_key = tuple(sorted((symbols[i_idx], symbols[j_idx])))
                pair_distances[pair_key].append(d)
                
            for pair in self.pairs:
                el1, el2 = pair
                distances = pair_distances.get(pair, [])
                h, _ = np.histogram(distances, bins=self.nbin, range=(0, self.rrange))
                n1 = np.sum(symbols == el1)
                n2 = np.sum(symbols == el2)
                if n1 == 0 or n2 == 0:
                    current_pdf = np.zeros(self.nbin)
                else:
                    if el1 == el2:
                        norm_factor = (n1 * (n1 - 1)) / (2.0 * volume)
                    else:
                        norm_factor = (n1 * n2) / volume
                    current_pdf = (h / self.volbin) / (norm_factor * 2)
                all_frame_data[pair].append(current_pdf)
        pdfs = np.zeros((len(self.pairs), self.nbin))

        for pair_ind, pair in enumerate(self.pairs):
            if all_frame_data[pair]:
                pdfs[pair_ind] = np.mean(all_frame_data[pair], axis=0)
            else:
                pdfs[pair_ind] = np.zeros(self.nbin)

        return pdfs

    def get_partial_pdf(self, pair: Tuple[str, str]) -> np.ndarray:
        """
        Get the partial probability density function (PDF) of a given pair of target atoms.

        Args:
            pair (Tuple[str, str]): A tuple of two elements representing the target atoms. Example: ('Si', 'O')

        Returns:
            np.ndarray: An array of shape (nbin,) containing the PDF values.
        """
        return self.partial_pdfs[self.pairs.index(pair)]

    def get_total_rdf(self, type: str = "neutron", broaden: Union[bool, int, float] = False) -> np.ndarray:
        """
        Calculate the total RDF for a given number of bins and range.

        Args:
            type (str, optional): The type of structure factor to calculate. Defaults to "neutron".
            broaden (Union[bool, int, float], optional): If True, apply Gaussian broadening to the RDF. 
                If a number, specify the maximum Q value for broadening. Defaults to False.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the total RDF values.
            
        Raises:
            ValueError: If type is invalid or broaden is invalid.
        """
        if type not in {"neutron", "xray", "approx_xray"}:
            raise ValueError("Invalid type. Choose either 'neutron', 'xray', or 'approx_xray'.")

        gr_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            pdf_val = self.get_partial_pdf(pair=pair)
            if type == "neutron":
                gr_tot = gr_tot + (self.timesby[ind] * pdf_val) / sum(self.timesby)
            elif type == "approx_xray":
                gr_tot = gr_tot + (self.approx_xray_timesby[ind] * pdf_val) / np.sum(self.approx_xray_timesby, axis=0)
            elif type == "xray":
#                denom_Q = np.sum(self.xray_cb)**2
#                for ind, pair in enumerate(self.pairs):
#                    numerator_Q = self.xray_timesby[ind]
#                    w_ij_Q = np.divide(numerator_Q, denom_Q, 
#                                    out=np.zeros_like(numerator_Q), 
#                                    where=denom_Q != 0)
#                    w_ij_eff = np.trapz(w_ij_Q, self.qval) / (self.qval[-1] - self.qval[0])
#                    gr_tot += w_ij_eff * pdf
                print(
                    " X-ray RDF using Fourier transform of xray scattering function f_ij(Q) is not implemented yet."
                )
                break
        if broaden:
            if isinstance(broaden, (int, float)) and not isinstance(broaden, bool): 
                # bool check needed because bool is subclass of int in Python
                Q_max = float(broaden)
            else:
                 raise ValueError("broaden must be a number (Q_max) to apply broadening.")
                 
            gr_tot = gaussian_broadening(gr_tot, self.xval, Q_max)

        return gr_tot

    def get_partial_structure_factor(self, target_atoms: Tuple[str, str], lorch: bool = False) -> np.ndarray:
        """
        Calculate the partial structure factor for a given target atoms within a specified range.

        Args:
            target_atoms (Tuple[str, str]): A tuple of two elements representing the target atoms.
            lorch (bool, optional): If True, apply Lorch correction to the structure factor.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the partial structure factor.
        """

        pdf_val = self.get_partial_pdf(pair=target_atoms)
        q_r = np.outer(self.qval, self.xval).T
        # Fix division by zero if xval contains 0 (it shouldn't based on init, but good to be safe)
        with np.errstate(divide='ignore', invalid='ignore'):
            q_r = np.sin(q_r) / q_r
            q_r[np.isnan(q_r)] = 1.0 # sin(0)/0 limit is 1
            
        A_q = np.ones((np.shape(self.qval)[0], 1, np.shape(self.xval)[0]))
        A_q = A_q * 4 * math.pi * self.xval**2 * (pdf_val - 1)
        A_q = np.moveaxis(A_q, 0, -1) * q_r
        if lorch:
            factor = np.pi*self.xval / self.rrange
            with np.errstate(divide='ignore', invalid='ignore'):
                lorch_correction = np.sin(factor) / factor
                lorch_correction[np.isnan(lorch_correction)] = 1.0
            A_q = A_q * lorch_correction
            
        A_q = 1 + self.aveden * np.trapz(A_q[0].T, self.xval)
        return A_q

    def get_structure_factor(self, type: str = "neutron", lorch: bool = False) -> np.ndarray:
        """
        Calculate the total structure factor.

        Args:
            type (str, optional): The type of structure factor to calculate. Defaults to "neutron".
            lorch (bool, optional): whether to apply lorch correction.

        Returns:
            np.ndarray: An array of shape (nbin,) containing the total structure factor.
        """
        if type not in {"neutron", "xray", "approx_xray"}:
            raise ValueError("Invalid type. Choose either 'neutron', 'xray'")

        S_q_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            partial_sq = self.get_partial_structure_factor(target_atoms=(pair[0], pair[1]), lorch=lorch)
            if type == "neutron":
                S_q_tot = S_q_tot + (self.timesby[ind] * partial_sq) / sum(self.timesby)
            elif type == "xray":
                S_q_tot = S_q_tot + (self.xray_timesby[ind] * partial_sq) / np.sum(self.xray_timesby, axis=0)
        return S_q_tot

    def get_T_r_pdf(self, type: str = "neutron", broaden: Union[bool, int, float] = False) -> np.ndarray:
        """
        Calculate the total correlation function T(r).
        
        T(r) = 4 * pi * r * rho_0 * g(r)
        where rho_0 is the average number density.

        Args:
            type (str, optional): The type of scattering ("neutron" or "xray"). Defaults to "neutron".
            broaden (Union[bool, int, float], optional): Broadening parameter. Defaults to False.

        Returns:
            np.ndarray: The T(r) function values.
        """
        return 4 * math.pi * self.xval * self.aveden * self.get_total_rdf(type=type, broaden=broaden)

    def get_reduced_pdf(self, type: str = "neutron", broaden: Union[bool, int, float] = False) -> np.ndarray:
        """
        Get reduced PDF G(r).
        """
        return (-4 * math.pi * self.xval * self.aveden) + (
            4 * math.pi * self.xval * self.aveden * self.get_total_rdf(type=type, broaden=broaden)
        )
    
    def get_N_running(self, pair: Tuple[str, str]) -> np.ndarray:
        """
        Calculate the running coordination number for a specific pair of elements.
        
        This is the integral of the partial RDF up to distance r:
        N(r) = Integral(4 * pi * rho_j * g_ij(r) * r^2 dr)

        Args:
            pair (Tuple[str, str]): Tuple of atomic symbols (e.g., ("Si", "O")).

        Returns:
            np.ndarray: The running coordination number as a function of r.
        """
        pair_pdf = self.get_partial_pdf(pair)
        n_v = len(np.where(self.chemical_symbols == pair[0])) / self.volume
        integrand = 4*np.pi*n_v*pair_pdf*self.xval**2
        return integrate.cumulative_trapezoid(integrand, self.xval, initial=0.0)

# Alias for backward compatibility
scattering = Scattering
