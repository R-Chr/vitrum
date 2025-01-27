# Implementation of rings is an adapted version of the code for calculating rings from https://github.com/MotokiShiga/sova-cui
# making the code compatible with the vitrum package.
# Its MIT License (MIT) can be seen below

# MIT License
#
# Copyright (c) 2024 Motoki Shiga
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import networkx as nx
from vitrum.glass_Atoms import glass_Atoms


def enumerate_guttman_ring(atoms_all, chemical_bond_index):
    """
    Enumerate all possible rings in a given set of atoms.

    Parameters:
        atoms_all (list): A list of Atoms objects representing the atoms in the system.
        chemical_bond_index (ndarray): An array of shape (n_bonds, 2) representing the indices of the atoms involved in each bond.

    Returns:
        set_rings (set): A set of tuples representing the indices of the atoms involved in each ring.
    """

    set_rings = set()
    G = nx.Graph()
    G.add_nodes_from(atoms_all)
    G.add_edges_from(chemical_bond_index)

    for i in range(chemical_bond_index.shape[0]):
        n0 = chemical_bond_index[i, 0]
        n1 = chemical_bond_index[i, 1]
        if (n0 in atoms_all) | (n1 in atoms_all):
            G.remove_edge(n0, n1)
            try:
                paths = nx.all_shortest_paths(G, source=n0, target=n1)
                for path in paths:
                    path = np.array(path)
                    i_min = np.argmin(path)
                    path = tuple(np.r_[path[i_min:], path[:i_min]])
                    # to aline the orientation
                    if path[-1] < path[1]:
                        path = path[::-1]
                        path = tuple(np.r_[path[-1], path[:-1]])
                    set_rings.add(path)
            except nx.NetworkXNoPath:
                pass  # Nothing to do
            G.add_edge(n0, n1)
    return set_rings


class Ring(object):
    """
    A class representing a ring in a atomistic system.

    Parameters:
        atoms (list): A list of Atoms objects representing the atoms in the system.
        indexes (list, optional): A list of indices of the atoms involved in the ring. Defaults to None.
    """

    def __init__(self, atoms, indexes=None):
        self.atoms = atoms
        self.indexes = indexes
        self._roundness = None
        self._roughness = None
        self.ellipsoid_lengths = None
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = [np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types]

    def center(self):
        """
        Calculate the center of the ring.

        Returns:
            center (ndarray): An array of shape (3,) representing the center of the ring.
        """
        positions = self.atoms.get_positions()
        cell = np.diagonal(self.atoms.get_cell())
        differences = np.diff(positions, axis=0, append=positions[:1])
        crossings = np.floor_divide(differences + 0.5 * cell, cell).astype(int)
        adjusted_positions = positions - np.cumsum(cell * crossings, axis=0)
        center = np.mean(adjusted_positions, axis=0)
        center = np.mod(center + cell, cell)
        return center

    def size(self):
        """
        Calculate the size of the ring, i.e., the number of atoms in the ring.

        Returns:
            size (int): The size of the ring.
        """
        return len(self.atoms)


class RINGs:
    """
    A class for calculating and analyzing rings in atomistic systems.

    Parameters:
        atoms (list): A list of Atoms objects representing the atoms in the system.
        included_atoms (list): A list of strings representing the chemical symbols of the atoms to include in the analysis.
        bonding_dict (dict): A dictionary mapping the chemical symbols of the atoms to their bonding partners.
    """

    def __init__(self, atoms, included_atoms, bonding_dict):
        super().__init__()
        self.bonding_dict = bonding_dict
        atoms = atoms[[atom.symbol in included_atoms for atom in atoms]]
        self.atoms = glass_Atoms(atoms)
        self.num_atoms = len(self.atoms)
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = [np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types]
        self.rings = []

    def get_bonds(self):
        """
        Get the bonds in the system.

        Returns:
            bonds (list): A list of tuples representing the indices of the atoms involved in each bond.
        """

        dist = self.atoms.get_dist()
        bonds = []
        for _, value in self.bonding_dict.items():
            atom_ids_1 = self.atom_ids[self.atom_types.index(value[0])]
            atom_ids_2 = self.atom_ids[self.atom_types.index(value[1])]
            cutoff = value[2]

            dist_list = dist[np.ix_(atom_ids_1, atom_ids_2)]

            for loc, id in enumerate(atom_ids_1):
                bond_ids = np.where((dist_list[loc, :] < cutoff) & (dist_list[loc, :] > 0))[0]

                for bond_id in bond_ids:
                    bonds.append([id, atom_ids_2[bond_id]])

        return bonds

    def calculate(self):
        """
        Calculate the rings in the system.

        Returns:
            rings (list): A list of Ring objects representing the rings in the system.
        """

        bonds = self.get_bonds()
        self.chemical_bond_index_atoms = np.array(bonds)
        atoms_all = np.arange(self.num_atoms)
        set_rings = enumerate_guttman_ring(atoms_all, self.chemical_bond_index_atoms)
        self.rings = [Ring(self.atoms[list(r)], list(r), self.bonding_dict) for r in list(set_rings)]
        return self.rings
