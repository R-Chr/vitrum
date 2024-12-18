import numpy as np
from vitrum.glass_Atoms import glass_Atoms
import itertools


class coordination:
    def __init__(self, atoms_list: list):
        """
        Initializes a new instance of the coordination class with a list of atoms.

        Parameters:
            atom_list (list): A list of atoms to be used for the coordination analysis.
        """

        self.atoms_list = [glass_Atoms(atom) for atom in atoms_list]
        self.chemical_symbols = atoms_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)

    def get_angle_distribution(self, center_type, neigh_types, nbin=70, cutoff="Auto", range=None):
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        Parameters:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (str, list): The atomic symbols of the neighbor atoms.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 70.
            cutoff (float, int, list, or "Auto", optional): Range within which to calculate the angular distribution.
              Defaults to "Auto". Can be a list of cutoffs for each neighbor type, or a specific cutoff for all.
            range (list, optional): The range of the histogram.
              Defaults to None (range is determined automatically from np.histogram).

        Returns:
            angles (ndarray): An array of shape (nbin,) containing the angle, values.
            dist (ndarray): A list of arrays containing the angular distribution values.
        """
        angles_all = []
        for atoms in self.atoms_list:
            angles = atoms.get_all_angles(center_type, neigh_types, cutoff)
            flat_list_angles = list(itertools.chain(*angles))
            angles_all.append(np.array(flat_list_angles))

        dist, edges = np.histogram(np.hstack(angles_all), bins=nbin, density=True, range=range)
        angles = edges[1:] - 0.5 * (np.ptp(edges) / nbin)
        return angles, dist

    def get_coordination_numbers(self, center_type, neigh_type, nbin=70, cutoff="Auto"):
        "Not implemented yet"
        pass
