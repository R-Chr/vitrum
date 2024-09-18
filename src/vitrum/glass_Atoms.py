import numpy as np
from ase import Atoms
from scipy.signal import argrelextrema
from typing import List, Union, Optional


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
        x_dif = np.abs(positions[:, 0][np.newaxis, :] - positions[:, 0][:, np.newaxis])
        y_dif = np.abs(positions[:, 1][np.newaxis, :] - positions[:, 1][:, np.newaxis])
        z_dif = np.abs(positions[:, 2][np.newaxis, :] - positions[:, 2][:, np.newaxis])
        x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
        y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
        z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
        i_i = np.sqrt(x_dif**2 + y_dif**2 + z_dif**2)
        return i_i

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

    def get_pdf(self, target_atoms, rrange=10, nbin=100):
        """
        Calculate the probability density function (PDF) of a given pair of target atoms within a specified range.

        Parameters:
            target_atoms (list): A list of two elements representing the target atoms. Each element can be either a string (chemical symbol) or an integer (atomic number).
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 10.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.

        Returns:
            xval (ndarray): An array of shape (nbin,) containing the distance values.
            pdf (ndarray): An array of shape (nbin,) containing the PDF values.
        """
        if isinstance(target_atoms[0], str):
            types = self.get_chemical_symbols()
        if isinstance(target_atoms[0], int):
            types = self.get_atomic_numbers()
        types = np.array(types)
        distances = self.get_dist()
        atom_1 = np.where(types == target_atoms[0])[0]
        atom_2 = np.where(types == target_atoms[1])[0]
        dist_list = distances[np.ix_(atom_1, atom_2)]
        edges = np.linspace(0, rrange, nbin + 1)
        xval = edges[1:] - 0.5 * (rrange / nbin)
        volbin = []
        for i in range(nbin):
            vol = ((4 / 3) * np.pi * (edges[i + 1]) ** 3) - ((4 / 3) * np.pi * (edges[i]) ** 3)
            volbin.append(vol)

        h, bin_edges = np.histogram(dist_list, bins=nbin, range=(0, rrange))
        h[0] = 0
        pdf = (h / volbin) / (dist_list.shape[0] * dist_list.shape[1] / self.get_volume())
        return xval, pdf

    def get_angular_dist(self, center_type, neigh_type, cutoff="Auto"):
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        Parameters:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (str): The atomic symbol of the neighbor atoms.
            cutoff (float, int, or "Auto", optional): The range within which to calculate the angular distribution. Defaults to "Auto".

        Returns:
            angles (list): A list of arrays containing the angular distribution values.
        """
        distances = self.get_dist()

        types = self.get_chemical_symbols()
        center_index = np.where(types == center_type)[0]
        neigh_index = np.where(types == neigh_type)[0]

        if cutoff == "Auto":
            pdf = self.get_pdf(target_atoms=[center_type, neigh_type])
            cutoff = pdf[0][self.find_min_after_peak(pdf[1])]
        elif isinstance(cutoff, float | int):
            cutoff = cutoff

        angles = []

        for center in center_index:
            neighbors = np.where((distances[neigh_index, center] < cutoff) & (distances[neigh_index, center] > 0))[0]
            if neighbors.shape[0] < 2:
                continue
            upper_index = np.triu_indices(neighbors.shape[0], k=1)
            comb_1 = np.meshgrid(neighbors, neighbors)[0][upper_index]
            comb_2 = np.meshgrid(neighbors, neighbors)[1][upper_index]
            indicies = np.vstack((neigh_index[comb_1], np.full(len(comb_1), center), neigh_index[comb_2])).T
            angles.append(self.get_angles(indicies, mic=True))
        return angles

    def get_coordination_number(self, center_type, neigh_type, cutoff="Auto"):
        """
        Calculate the coordination number of a given pair of target atoms within a specified range.

        Parameters:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (str): The atomic symbol of the neighbor atoms.
            cutoff (float, int, or "Auto", optional): The range within which to calculate the coordination number. Defaults to "Auto".

        Returns:
            coordination_numbers (list): A list containing the coordination numbers.
        """

        distances = self.get_dist()
        types = self.get_chemical_symbols()
        atom_1 = np.where(types == center_type)[0]
        atom_2 = np.where(types == neigh_type)[0]
        dist_list = distances[np.ix_(atom_1, atom_2)]

        if cutoff == "Auto":
            pdf = self.get_pdf(target_atoms=[center_type, neigh_type])
            cutoff = pdf[0][self.find_min_after_peak(pdf[1])]
        elif isinstance(cutoff, float | int):
            cutoff = cutoff
        print(cutoff)
        coordination_numbers = []
        for center in range(len(atom_1)):
            neighbors = np.where((dist_list[center, :] < cutoff) & (dist_list[center, :] > 0))[0]
            coordination_numbers.append(neighbors.shape[0])
        return coordination_numbers

    def find_min_after_peak(self, padf):
        mins = argrelextrema(padf, np.less_equal, order=4)[0]
        second_min = [i for ind, i in enumerate(mins) if i != ind][0]
        return second_min

    def NBO_analysis(
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
            former_types (Optional[List[str]], optional): A list of types of the former atoms. Defaults to None.
            cutoff (Union[str, float, int], optional): The cutoff distance for considering a bridge.
                If "Auto", the cutoff is determined by finding the minimum value after the peak in the
                radial distribution function. If a float or int, the cutoff is set to the specified value.
                Defaults to "Auto".

        Returns:
            List[int]: A list of the number of bridges for each center atom.
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
            cutoff = pdf[0][self.find_min_after_peak(pdf[1])]
        elif isinstance(cutoff, (float, int)):
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
