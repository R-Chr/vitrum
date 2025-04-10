import numpy as np
from ase import Atoms
from typing import List, Union, Optional
from vitrum.utility import pdf, find_min_after_peak
from vitrum.utility import get_dist as dist
from itertools import product


class glass_Atoms(Atoms):
    # def __init__(self):
    #    super().__init__()

    def get_dist(self):
        """
        Calculate the distances between all pairs of atoms in the Atoms object.

        Returns:
            i_i (ndarray): An array of shape (n_atoms, n_atoms) containing the distances
                between each pair of atoms.
        """
        dim = np.diagonal(self.get_cell())
        positions = self.get_positions()
        return dist(positions, dim)

    def set_new_chemical_symbols(self, dict):
        """
        Set new chemical symbols for the atoms in the object.

        Parameters:
            dict (dict): A dictionary mapping atomic numbers to new chemical symbols.

        Returns:
            None
        """
        corr_symbols = [dict[i] for i in self.get_atomic_numbers()]
        self.set_chemical_symbols(corr_symbols)

    def get_pdf(self, target_atoms, rrange=10, nbin=100, indicies=None):
        """
        Calculate the probability density function (PDF) of a given pair of target atoms within a specified range.

        Parameters:
            target_atoms (list): A list of two elements representing the target atoms. Each element can be either a
              string (chemical symbol) or an integer (atomic number).
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 10.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.
            indicies (list, optional): A list of two elements representing the indicies of the target atoms. Specifying
              this parameter will override the target_atoms parameter. Defaults to None.

        Returns:
            xval (ndarray): An array of shape (nbin,) containing the distance values.
            pdf (ndarray): An array of shape (nbin,) containing the PDF values.
        """
        if indicies is None:
            if isinstance(target_atoms[0], str):
                types = self.get_chemical_symbols()
            if isinstance(target_atoms[0], int):
                types = self.get_atomic_numbers()
            types = np.array(types)
            distances = self.get_dist()
            atom_1 = np.where(types == target_atoms[0])[0]
            atom_2 = np.where(types == target_atoms[1])[0]
        else:
            atom_1 = indicies[0]
            atom_2 = indicies[1]
        dist_list = distances[np.ix_(atom_1, atom_2)]
        return pdf(dist_list, self.get_volume(), rrange, nbin)

    def get_all_angles(self, center_type, neigh_types, cutoff="Auto"):
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        Parameters:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (str, list): The atomic symbols of the neighbor atoms.
            cutoff (float, int, list, or "Auto", optional): Range within which to calculate the angular distribution.
              Defaults to "Auto". Can be a list of cutoffs for each neighbor type, or a specific cutoff for all.

        Returns:
            angles (list): A list of arrays containing the angular distribution values.
        """
        types = np.array(self.get_chemical_symbols())
        species = np.unique(types)

        if isinstance(neigh_types, str):
            neigh_types = [neigh_types, neigh_types]

        if center_type not in species:
            raise ValueError(f"The center type {center_type} is not in the list of species.")
        if neigh_types[0] not in species or neigh_types[1] not in species:
            raise ValueError(f"The neighbor types {neigh_types[0]} or {neigh_types[1]} is not in the list of species.")

        distances = self.get_dist()

        center_index = np.where(types == center_type)[0]
        neigh_index = [np.where(types == neigh_type)[0] for neigh_type in neigh_types]

        if cutoff == "Auto":
            pdf = [self.get_pdf(target_atoms=[center_type, neigh_type]) for neigh_type in neigh_types]
            cutoff = [pdf[i][0][find_min_after_peak(pdf[i][1])] for i in range(len(neigh_types))]
        elif isinstance(cutoff, float) or isinstance(cutoff, int):
            cutoff = [cutoff, cutoff]

        angles = []

        for center in center_index:

            neighbor1 = np.where(
                (distances[neigh_index[0], center] < cutoff[0]) & (distances[neigh_index[0], center] > 0)
            )[0]

            neighbor2 = np.where(
                (distances[neigh_index[1], center] < cutoff[1]) & (distances[neigh_index[1], center] > 0)
            )[0]

            unique_pairs = set()

            # Generate combinations
            for a, b in product(neigh_index[0][neighbor1], neigh_index[1][neighbor2]):
                if a != b:
                    unique_pairs.add(tuple(sorted((a, b))))
            combinations = np.array(list(unique_pairs))
            if combinations.shape[0] < 1:
                continue

            indicies = np.vstack((combinations[:, 0], np.full(len(combinations), center), combinations[:, 1])).T
            angles.append(self.get_angles(indicies, mic=True))
        return angles

    def get_coordination_number(self, center_type, neigh_type, cutoff="Auto"):
        """
        Calculate the coordination number of a given pair of target atoms within a specified range.

        Parameters:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (str): The atomic symbol of the neighbor atoms.
            cutoff (float, int, or "Auto", optional): The range within which to calculate the coordination number.
              Defaults to "Auto".

        Returns:
            coordination_numbers (list): A list containing the coordination numbers.
        """

        distances = self.get_dist()
        types = np.array(self.get_chemical_symbols())
        atom_1 = np.where(types == center_type)[0]
        atom_2 = np.where(types == neigh_type)[0]
        dist_list = distances[np.ix_(atom_1, atom_2)]

        if cutoff == "Auto":
            pdf = self.get_pdf(target_atoms=[center_type, neigh_type])
            cutoff = pdf[0][find_min_after_peak(pdf[1])]
        elif isinstance(cutoff, float) or isinstance(cutoff, int):
            cutoff = cutoff
        coordination_numbers = []
        for center in range(len(atom_1)):
            neighbors = np.where((dist_list[center, :] < cutoff) & (dist_list[center, :] > 0))[0]
            coordination_numbers.append(neighbors.shape[0])
        return coordination_numbers

    def get_bridging_analysis(
        self,
        center_type: str,
        bridge_type: str,
        former_types: Optional[List[str]] = None,
        cutoff: Union[float, int] = "Auto",
    ) -> List[int]:
        """
        Calculate the number of bridges for each center atom of a given type.

        Parameters:
            center_type (str): The type of the center atoms.
            bridge_type (str): The type of the bridge atoms.
            former_types (Optional[List], optional): A list of types of the former atoms. Defaults to None.
            cutoff (Union[str, float, int], optional): The cutoff distance for considering a bridge.
                If "Auto", the cutoff is determined by finding the minimum value after the peak in the
                radial distribution function. If a float or int, the cutoff is set to the specified value.
                Defaults to "Auto".

        Returns:
            num_of_bridges (list): A list of the number of bridges for each center atom.
        """
        distances = self.get_dist()
        types = np.array(self.get_chemical_symbols())
        centers = np.where(types == center_type)[0]
        bridges = np.where(types == bridge_type)[0]
        dist_list = distances[np.ix_(centers, bridges)]

        if former_types is None:
            formers = centers
        elif isinstance(former_types, list):
            formers = np.hstack([np.where(types == type)[0] for type in former_types])
        else:
            raise TypeError("former_types must be either None or a List of atom types")

        if cutoff == "Auto":
            pdf = self.get_pdf(target_atoms=[center_type, bridge_type])
            cutoff = pdf[0][find_min_after_peak(pdf[1])]
        elif isinstance(cutoff, float) or isinstance(cutoff, int):
            cutoff = cutoff

        num_of_bridges = []
        for center in centers:
            neighbors = np.where((dist_list[center, :] < cutoff) & (dist_list[center, :] > 0))[0]
            neighbors = [bridges[neighbor] for neighbor in neighbors]
            q_species = 0
            for neigh in neighbors:
                num_bridges = np.where((distances[formers, neigh] < cutoff) & (distances[formers, neigh] > 0))[0]
                if num_bridges.shape[0] >= 2:
                    q_species += 1
            num_of_bridges.append(q_species)

        return num_of_bridges

    def get_density(self):
        return (np.sum(self.get_masses()) / 6.02214076 * 10**-23) / (self.get_volume() * 10**-24)
