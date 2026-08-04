import itertools
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from scipy import integrate
from scipy.stats import norm
from tqdm import tqdm

from vitrum.geometry import pdf, radial_bins, require_orthorhombic
from vitrum.glass_atoms import GlassAtoms


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
    dist_broad = np.trapezoid(foubroad, r)
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

        cell_lengths = require_orthorhombic(atom_list[0].get_cell(), "Scattering")
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
        self.xval, self.volbin = radial_bins(self.rrange, self.nbin)

        self.qval = np.linspace(qmin, qmax, self.nbin)
        self.chemical_symbols = atom_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.pairs = [pair for pair in itertools.product(self.species, repeat=2)]
        self.c = [self.chemical_symbols.count(i) / len(self.chemical_symbols) for i in self.species]

        # The composition is assumed constant across the trajectory: self.chemical_symbols,
        # self.species, self.pairs and self.c are all derived from the first frame alone.
        reference_composition = sorted(self.chemical_symbols)
        for frame_ind, atom in enumerate(atom_list[1:], start=1):
            if sorted(atom.get_chemical_symbols()) != reference_composition:
                raise ValueError(
                    f"Frame {frame_ind} has a different composition to frame 0. Scattering "
                    "assumes a fixed composition across the trajectory."
                )

        # Trajectory averages, so that NPT runs with a varying cell are weighted consistently
        # with the per-frame partial PDFs.
        volumes = np.array([atom.get_volume() for atom in atom_list])
        self.volume = float(volumes.mean())
        self.aveden = float(np.mean([len(atom) / vol for atom, vol in zip(atom_list, volumes)]))
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
        pdf_sum = np.zeros((len(self.pairs), self.nbin))
        n_frames = len(self.atom_list)

        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            distances = atom.get_dist()
            symbols = np.array(atom.get_chemical_symbols())
            volume = atom.get_volume()

            for pair_ind, pair in enumerate(self.pairs):
                idx_1 = np.flatnonzero(symbols == pair[0])
                idx_2 = np.flatnonzero(symbols == pair[1])
                like_pair = pair[0] == pair[1]
                if like_pair:
                    # Exclude self-pairs: N_a atoms each have N_a - 1 distinct partners.
                    n_pairs = len(idx_1) * (len(idx_1) - 1)
                else:
                    n_pairs = len(idx_1) * len(idx_2)
                dist_list = distances[np.ix_(idx_1, idx_2)]
                _, current_pdf = pdf(
                    dist_list,
                    volume,
                    self.rrange,
                    self.nbin,
                    n_pairs=n_pairs,
                    exclude_self=like_pair,
                )
                pdf_sum[pair_ind, :] += current_pdf
        return pdf_sum / n_frames


    def calculate_partial_pdfs_neighborhood(self):
        """
        Calculate partial PDFs using O(N) neighbor lists

        Returns:
            np.ndarray: Array of partial PDFs.
        """
        all_frame_data = {pair: [] for pair in self.pairs}
        for atom in tqdm(self.atom_list, disable=self.disable_progress):
            symbols = np.array(atom.get_chemical_symbols())
            volume = atom.get_volume()
            i_list, j_list, d_list = neighbor_list("ijd", a=atom, cutoff=self.rrange)
            pair_distances = defaultdict(list)
            for i_idx, j_idx, d in zip(i_list, j_list, d_list):
                pair_key = tuple(sorted((symbols[i_idx], symbols[j_idx])))
                pair_distances[pair_key].append(d)

            for pair in self.pairs:
                el1, el2 = pair
                # Distances are keyed by a sorted element tuple, so the lookup key must be
                # sorted too; g_ij == g_ji, so both orderings of a cross pair share a key.
                distances = pair_distances.get(tuple(sorted(pair)), [])
                n1 = int(np.sum(symbols == el1))
                n2 = int(np.sum(symbols == el2))
                # `neighbor_list` reports both (i, j) and (j, i), so a cross pair's distances
                # appear twice. A like pair's n1 * (n1 - 1) ordered pairs already account for it.
                if el1 == el2:
                    n_pairs = n1 * (n1 - 1)
                else:
                    n_pairs = 2 * n1 * n2
                _, current_pdf = pdf(
                    distances,
                    volume,
                    self.rrange,
                    self.nbin,
                    n_pairs=n_pairs,
                    exclude_self=False,  # a neighbour list never contains self-distances
                )
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
            NotImplementedError: If type is "xray"; use "approx_xray" instead.
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
                raise NotImplementedError(
                    "X-ray RDF using the Fourier transform of the x-ray scattering function "
                    "f_ij(Q) is not implemented. Use type='approx_xray' for the Q-independent "
                    "atomic-number approximation, or type='neutron'."
                )
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
            
        A_q = 1 + self.aveden * np.trapezoid(A_q[0].T, self.xval)
        return A_q

    def get_weighted_partial_structure_factors(
        self,
        type: str = "neutron",
        lorch: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate weighted partial structure factors W_ij * S_ij(Q) for all
        unique element pairs.

        Weights follow the same definition as get_structure_factor():
            neutron: W_ij = c_i*b_i * c_j*b_j / (sum_k c_k*b_k)^2
            xray:    W_ij(Q) = c_i*f_i(Q) * c_j*f_j(Q) / (sum_k c_k*f_k(Q))^2

        Cross terms (i != j) are merged: S_ij = S_ji, so their weights are
        multiplied by 2 and only one label (e.g. "Si-O") is returned.

        Summing the returned values gives get_structure_factor(type=type).

        Args:
            type (str): Weighting scheme, "neutron" or "xray". Defaults to "neutron".
            lorch (bool): If True, apply Lorch modification function to reduce
                truncation ripples. Passed through to get_partial_structure_factor().
                Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dict mapping pair label (e.g. "Si-O") to
                W_ij * S_ij(Q), each of shape (nbin,).

        Raises:
            ValueError: If type is not "neutron" or "xray".
        """
        if type == "neutron":
            denom = sum(self.timesby)
        elif type == "xray":
            denom = np.sum(self.xray_timesby, axis=0)
        else:
            raise ValueError("Invalid type. Choose either 'neutron' or 'xray'.")

        unique_pairs = list(itertools.combinations_with_replacement(self.species, 2))

        weighted_partials: Dict[str, np.ndarray] = {}

        for pair in unique_pairs:
            label = f"{pair[0]}-{pair[1]}"

            try:
                idx = self.pairs.index(pair)
            except ValueError:
                idx = self.pairs.index((pair[1], pair[0]))

            partial_sq = self.get_partial_structure_factor(
                target_atoms=(pair[0], pair[1]), lorch=lorch
            )

            multiplier = 1.0 if pair[0] == pair[1] else 2.0
            if type == "neutron":
                weight = multiplier * self.timesby[idx] / denom
            else:
                weight = multiplier * self.xray_timesby[idx] / denom

            w_sij = np.asarray(weight * partial_sq, dtype=float)
            weighted_partials[label] = w_sij

        return weighted_partials


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
            elif type == "approx_xray":
                S_q_tot = S_q_tot + (self.approx_xray_timesby[ind] * partial_sq) / np.sum(self.approx_xray_timesby, axis=0)
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
        # rho_j is the density of the neighbour species, pair[1]. Expressed as concentration x
        # average density so it holds for NPT trajectories.
        c_j = self.c[list(self.species).index(pair[1])]
        n_v = c_j * self.aveden
        integrand = 4*np.pi*n_v*pair_pdf*self.xval**2
        return integrate.cumulative_trapezoid(integrand, self.xval, initial=0.0)
