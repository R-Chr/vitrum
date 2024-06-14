import pandas as pd
import numpy as np
import itertools
import math
from pathlib import Path


class scattering:
    def __init__(self, atom_list):
        """
        Initializes a new instance of the class with the given atom_list.

        Parameters:
            atom_list (list): A list of Atoms objects representing the atom list.

        Returns:
            None
        """
        self.atom_list = atom_list
        script_dir = Path(__file__).parent
        self.scattering_lengths = pd.read_csv(script_dir / "scattering_lengths.csv", sep=";", decimal=",")

        self.chemical_symbols = atom_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.pairs = itertools.product(self.species, repeat=2)

        self.b = np.array(
            [self.scattering_lengths[self.scattering_lengths["Isotope"] == i]["b"] for i in self.species]
        ).flatten()
        self.c = [self.chemical_symbols.count(i) / len(self.chemical_symbols) for i in self.species]
        self.cb = [i * j for i, j in zip(self.c, self.b)]
        self.timesby = [pair[0] * pair[1] for pair in itertools.product(self.cb, repeat=2)]

        self.aveden = len(atom_list[0]) / atom_list[0].get_volume()

    def get_partial_pdf(self, pair, rrange=15, nbin=100):
        """
        Calculate the partial probability density function (PDF) of a
          given pair of target atoms within a specified range.

        Parameters:
            pair (list): A list of two elements representing the target atoms.
              Each element can be either a string (chemical symbol) or an integer (atomic number).
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 15.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.

        Returns:
            xval (ndarray): An array of shape (nbin,) containing the distance values.
            pdf (ndarray): An array of shape (nbin,) containing the PDF values.
        """
        pdf = np.mean(
            [atoms.get_pdf(target_atoms=[pair[0], pair[1]], rrange=rrange, nbin=nbin)[1] for atoms in self.atom_list],
            axis=0,
        )
        xval = self.atom_list[0].get_pdf(target_atoms=[pair[0], pair[1]], rrange=rrange, nbin=nbin)[0]
        return xval, pdf

    def get_total_rdf(self, nbin=100, rrange=15):
        """
        Calculate the total RDF for a given number of bins and range.

        Parameters:
            nbin (int): The number of bins to use for the RDF calculation. Default is 100.
            rrange (int): The range of distances to consider for the RDF calculation. Default is 15.

        Returns:
            tuple: A tuple containing the x-axis values of the RDF plot and the corresponding y-axis values.
        """
        gr_tot = np.zeros([nbin])
        for ind, pair in enumerate(self.pairs):
            pdf = self.get_partial_pdf(pair=pair, rrange=rrange, nbin=nbin)
            gr_tot = gr_tot + (self.timesby[ind] * pdf[1]) / sum(self.timesby)
        return pdf[0], gr_tot

    def get_partial_structure_factor(self, target_atoms: list, qrange=30, nbin=100, rrange=15):
        """
        Calculate the partial structure factor for a given target atoms within a specified range.

        Parameters:
            target_atoms (list): A list of two elements representing the target atoms.
              Each element can be either a string (chemical symbol) or an integer (atomic number).
            qrange (float, optional): The range within which to calculate the structure factor. Defaults to 30.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 15.

        Returns:
            tuple: A tuple containing the q-values (qval) and the partial structure factor
              (A_q) for the given target atoms.
                - qval (ndarray): An array of shape (nbin,) containing the q-values.
                - A_q (ndarray): An array of shape (nbin, 1, nbin) containing the partial structure factor.
        """

        qval = np.linspace(0.5, qrange, nbin)
        xval, pdf = self.get_partial_pdf(pair=target_atoms, rrange=rrange, nbin=nbin)
        q_r = np.outer(qval, xval).T
        q_r = np.sin(q_r) / q_r
        A_q = np.ones((np.shape(qval)[0], 1, np.shape(xval)[0]))
        A_q = A_q * 4 * math.pi * xval**2 * (pdf - 1)
        A_q = np.moveaxis(A_q, 0, -1) * q_r
        A_q = 1 + self.aveden * np.trapz(A_q[0].T, xval)
        return qval, A_q

    def get_structure_factor(self, nbin=100, rrange=15, qrange=30):
        """
        Calculate the total structure factor for a given number of bins and range.

        Parameters:
            nbin (int, optional): The number of bins to use for the structure factor calculation. Defaults to 100.
            rrange (float, optional): The range within which to calculate the structure factor. Defaults to 15.
            qrange (float, optional): The range within which to calculate the structure factor. Defaults to 30.

        Returns:
            tuple: A tuple containing the q-values (qval) and the total structure factor
              (S_q_tot) for the given number of bins and range.
                - qval (ndarray): An array of shape (nbin,) containing the q-values.
                - S_q_tot (ndarray): An array of shape (nbin,) containing the total structure factor.
        """
        S_q_tot = np.zeros(nbin)
        for ind, pair in enumerate(self.pairs):
            qval, partial_sq = self.get_partial_structure_factor(
                target_atoms=[pair[0], pair[1]], nbin=nbin, rrange=rrange, qrange=qrange
            )
            S_q_tot = S_q_tot + (self.timesby[ind] * partial_sq) / sum(self.timesby)
        return qval, S_q_tot
