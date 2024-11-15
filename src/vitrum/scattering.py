import pandas as pd
import numpy as np
import itertools
import math
from pathlib import Path
from vitrum.glass_Atoms import glass_Atoms
from tqdm import tqdm


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

        f_i = []

        for ind in range(len(self.species)):

            f_i.append(
                np.sum(
                    [
                        self.x_ray_a[ind][i] * np.exp(-self.x_ray_b[ind][i] * ((self.qval) / (4 * np.pi)) ** 2)
                        for i in range(4)
                    ],
                    axis=0,
                )
                + self.x_ray_c[ind]
            )

        self.xray_cb = [i * j for i, j in zip(self.c, f_i)]
        self.xray_timesby = [pair[0] * pair[1] for pair in itertools.product(self.xray_cb, repeat=2)]

    def get_partial_pdf(self, pair):
        """
        Calculate the partial probability density function (PDF) of a
          given pair of target atoms within a specified range.

        Parameters:
            pair (list): A list of two elements representing the target atoms.
              Each element can be either a string (chemical symbol) or an integer (atomic number).

        Returns:
            pdf (ndarray): An array of shape (nbin,) containing the PDF values.
        """
        print("Calculating RDFs...")
        pdf = np.mean(
            [
                atoms.get_pdf(target_atoms=[pair[0], pair[1]], rrange=self.rrange, nbin=self.nbin)[1]
                for atoms in tqdm(self.atom_list)
            ],
            axis=0,
        )
        return pdf

    def get_total_rdf(self, type="neutron"):
        """
        Calculate the total RDF for a given number of bins and range.

        Parameters:
            type (str, optional): The type of structure factor to calculate. Defaults to "neutron".

        Returns:
            gr_tot (ndarray): An array of shape (nbin,) containing the total RDF values.
        """
        gr_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            pdf = self.get_partial_pdf(pair=pair)
            if type == "neutron":
                gr_tot = gr_tot + (self.timesby[ind] * pdf) / sum(self.timesby)
            elif type == "fake_xray":
                pass
            elif type == "xray":
                gr_tot = gr_tot + (self.xray_timesby[ind] * pdf) / np.sum(self.xray_timesby, axis=0)
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
        S_q_tot = np.zeros(self.nbin)
        for ind, pair in enumerate(self.pairs):
            partial_sq = self.get_partial_structure_factor(target_atoms=[pair[0], pair[1]])
            if type == "neutron":
                S_q_tot = S_q_tot + (self.timesby[ind] * partial_sq) / sum(self.timesby)
            elif type == "fake_xray":
                pass
            elif type == "xray":
                S_q_tot = S_q_tot + (self.xray_timesby[ind] * partial_sq) / np.sum(self.xray_timesby, axis=0)
        return S_q_tot
