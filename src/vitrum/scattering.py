import pandas as pd
import numpy as np
import itertools
import math
from pathlib import Path
from vitrum.glass_Atoms import glass_Atoms
from tqdm import tqdm
from scipy.stats import norm
from ase import Atom


def gaussian_broadening(g_r, r, Q_max):
    # Broad by using a convolution using a Gaussian function
    delta_r = r[np.newaxis, :] - r[:, np.newaxis]
    sum_r = r[np.newaxis, :] + r[:, np.newaxis]
    FWHM = 5.437 / Q_max
    sigma = FWHM / 2.355
    foubroad = g_r * (norm.pdf(delta_r, 0, sigma) - norm.pdf(sum_r, 0, sigma))
    dist_broad = np.trapz(foubroad, r)
    return dist_broad


class scattering:
    def __init__(
        self, atom_list, qrange=30, rrange=15, nbin=500, neutron_scattering_coef=None, x_ray_scattering_coef=None
    ):
        """
        Initializes a new instance of the class with the given atom_list.

        Parameters:
            atom_list (list): A list of Atoms objects representing the atom list.
            qrange (float, optional): The range of q-values to use. Defaults to 30.
            rrange (float, optional): The range of r-values to use. Defaults to 15.
            nbin (int, optional): The number of bins to use. Defaults to 500.
            neutron_scattering_coef (list, optional): A list of custom neutron scattering lengths. Defaults to None.
              If None, the default coefficients Neutron News, Vol. 3, No. 3, 1992, pp. 29-37 are used.
            x_ray_scattering_coef (list, optional): A list of custom x-ray scattering coefficients. Defaults to None.
              If None, the default coefficients from International Tables for Crystallography (2006). Vol. C. ch. 6.1,
              pp. 554-595 https://doi.org/10.1107/97809553602060000600 Table 6.1.1.4 are used.

        Returns:
            None
        """
        self.atom_list = [glass_Atoms(atom) for atom in atom_list]
        script_dir = Path(__file__).parent

        self.rrange = rrange
        self.nbin = nbin
        edges = np.linspace(0, self.rrange, self.nbin + 1)
        self.xval = edges[1:] - 0.5 * (self.rrange / self.nbin)
        self.qval = np.linspace(0.5, qrange, self.nbin)
        self.chemical_symbols = atom_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.pairs = [pair for pair in itertools.product(self.species, repeat=2)]
        self.c = [self.chemical_symbols.count(i) / len(self.chemical_symbols) for i in self.species]
        self.aveden = len(atom_list[0]) / atom_list[0].get_volume()
        self.atomic_numbers = [Atom(atom).number for atom in self.species]

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
            x_ray_scattering_coef = pd.read_csv(script_dir / "x_ray_scattering_factor_coefficients.csv", sep=",")
            x_ray_scattering_coef = np.array(
                [x_ray_scattering_coef[x_ray_scattering_coef["Element"] == i] for i in self.species]
            ).reshape([len(self.species), 10])

        self.x_ray_a = x_ray_scattering_coef[:, [1, 3, 5, 7]]
        self.x_ray_b = x_ray_scattering_coef[:, [2, 4, 6, 8]]
        self.x_ray_c = x_ray_scattering_coef[:, [9]]

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

        self.partial_pdfs = self.calculate_partial_pdfs()

    def calculate_partial_pdfs(self):
        pdfs = np.zeros((len(self.atom_list), len(self.pairs), self.nbin))
        for atom_ind, atom in enumerate(tqdm(self.atom_list)):
            distances = atom.get_dist()
            for pair_ind, pair in enumerate(self.pairs):
                atom_1 = np.where(np.array(atom.get_chemical_symbols()) == pair[0])[0]
                atom_2 = np.where(np.array(atom.get_chemical_symbols()) == pair[1])[0]
                dist_list = distances[np.ix_(atom_1, atom_2)]
                edges = np.linspace(0, self.rrange, self.nbin + 1)
                volbin = []
                for i in range(self.nbin):
                    vol = ((4 / 3) * np.pi * (edges[i + 1]) ** 3) - ((4 / 3) * np.pi * (edges[i]) ** 3)
                    volbin.append(vol)
                h, bin_edges = np.histogram(dist_list, bins=self.nbin, range=(0, self.rrange))
                h[0] = 0
                pdfs[atom_ind, pair_ind, :] = (h / volbin) / (
                    dist_list.shape[0] * dist_list.shape[1] / atom.get_volume()
                )
        pdfs = np.mean(pdfs, axis=0)
        return pdfs

    def get_partial_pdf(self, pair: tuple):
        """
        get the partial probability density function (PDF) of a
          given pair of target atoms

        Parameters:
            pair (tuple): A tuple of two elements representing the target atoms. Example: ('Si', 'O')

        Returns:
            pdf (ndarray): An array of shape (nbin,) containing the PDF values.
        """
        return self.partial_pdfs[self.pairs.index(pair)]

    def get_total_rdf(self, type: str = "neutron", broaden: bool | int | float = False):
        """
        Calculate the total RDF for a given number of bins and range.

        Parameters:
            type (str, optional): The type of structure factor to calculate. Defaults to "neutron".
            broaden (bool | int | float, optional): If True, apply Gaussian broadening to the RDF. If a number, specify the maximum Q value
              for broadening. Defaults to False.

        Returns:
            gr_tot (ndarray): An array of shape (nbin,) containing the total RDF values.
        """
        if type not in {"neutron", "xray", "approx_xray"}:
            raise ValueError("Invalid type. Choose either 'neutron', 'xray', or 'approx_xray'.")

        gr_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            pdf = self.get_partial_pdf(pair=pair)
            if type == "neutron":
                gr_tot = gr_tot + (self.timesby[ind] * pdf) / sum(self.timesby)
            elif type == "approx_xray":
                gr_tot = gr_tot + (self.approx_xray_timesby[ind] * pdf) / np.sum(self.approx_xray_timesby, axis=0)
            elif type == "xray":

                ### Attempt to use Fourier transform of xray scattering function f_ij(Q)
                #  xray_cb = np.array([i * j for i, j in zip(self.c, self.f_i)])
                #  xray_timesby = np.array([pair[0] * pair[1] for pair in itertools.product(self.f_i, repeat=2)])
                #  f_ij = xray_timesby / np.sum(xray_cb, axis=0)
                #  g_x_all = []
                #  for f_i, pdf in zip(f_ij, pdfs):
                #      cos_qr = np.cos(np.outer(self.xval, self.qval))  # shape (Q, r)
                #      j_ij = (1 / np.pi) * np.trapz(f_i * cos_qr, self.qval, axis=0)  # shape (r,)
                #      y = sc.xval
                #      f_y = y * (pdf-1)
                #      conv_result = np.convolve(f_y, j_ij, mode='same') * (y[1] - y[0])  # scale by dy
                #      g_X= conv_result / y
                #      g_x_all.append(g_X)
                #  cc = np.array([[pair[0] * pair[1] for pair in itertools.product(self.c, repeat=2)]]).T
                #  gr_tot = np.sum(cc*np.array(g_x_all), axis=0)

                print(
                    " X-ray RDF using Fourier transform of xray scattering function f_ij(Q) is not implemented yet. Using approximation from atomic number."
                )
                # Use the approximate X-ray scattering factor
                gr_tot = gr_tot + (self.approx_xray_timesby[ind] * pdf) / np.sum(self.approx_xray_timesby, axis=0)

        if broaden:
            if isinstance(broaden, (int, float)):
                Q_max = broaden
            else:
                raise ValueError("broaden must be a number or False")
            gr_tot = gaussian_broadening(gr_tot, self.xval, Q_max)

        return gr_tot

    def get_partial_structure_factor(self, target_atoms: list):
        """
        Calculate the partial structure factor for a given target atoms within a specified range.

        Parameters:
            target_atoms (list): A list of two elements representing the target atoms.
              Each element can be either a string (chemical symbol) or an integer (atomic number).

        Returns:
            A_q (ndarray): An array of shape (nbin, 1, nbin) containing the partial structure factor.
        """

        pdf = self.get_partial_pdf(pair=target_atoms)
        q_r = np.outer(self.qval, self.xval).T
        q_r = np.sin(q_r) / q_r
        A_q = np.ones((np.shape(self.qval)[0], 1, np.shape(self.xval)[0]))
        A_q = A_q * 4 * math.pi * self.xval**2 * (pdf - 1)
        A_q = np.moveaxis(A_q, 0, -1) * q_r
        A_q = 1 + self.aveden * np.trapz(A_q[0].T, self.xval)
        return A_q

    def get_structure_factor(self, type="neutron"):
        """
        Calculate the total structure factor for a given number of bins and range.

        Parameters:
            type (str, optional): The type of structure factor to calculate. Defaults to "neutron".

        Returns:
            S_q_tot (ndarray): An array of shape (nbin,) containing the total structure factor.
        """
        if type not in {"neutron", "xray", "approx_xray"}:
            raise ValueError("Invalid type. Choose either 'neutron', 'xray', or 'approx_xray'.")

        S_q_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            partial_sq = self.get_partial_structure_factor(target_atoms=(pair[0], pair[1]))
            if type == "neutron":
                S_q_tot = S_q_tot + (self.timesby[ind] * partial_sq) / sum(self.timesby)
            elif type == "approx_xray":
                raise NotImplementedError("Approximate X-ray structure factor is not implemented.")
            elif type == "xray":
                S_q_tot = S_q_tot + (self.xray_timesby[ind] * partial_sq) / np.sum(self.xray_timesby, axis=0)
        return S_q_tot

    def get_T_r_pdf(self, type="neutron", broaden: bool | int | float = False):
        return 4 * math.pi * self.xval * self.aveden * self.get_total_rdf(type=type, broaden=broaden)

    def get_reducded_pdf(self, type="neutron", broaden: bool | int | float = False):
        return (-4 * math.pi * self.xval * self.aveden) + (
            4 * math.pi * self.xval * self.aveden * self.get_total_rdf(type=type, broaden=broaden)
        )


# Make pdf from distance list into its own function q
# def pdf(dist_list, atoms, rrange=10, nbin=100):
#    edges = np.linspace(0,rrange,nbin+1)
#    xval=edges[1:]-0.5*(rrange/nbin)
#    volbin = []
#    for i in range(nbin):
#        vol = ((4/3)*np.pi*(edges[i+1])**3)-((4/3)*np.pi*(edges[i])**3)
#        volbin.append(vol)
#
#    h, bin_edges = np.histogram(dist_list, bins=nbin, range=(0,rrange))
#    h[0] = 0
#    pdf = (h/volbin)/(dist_list.shape[0]*dist_list.shape[1]/atoms.get_volume())
#    return xval, pdf
