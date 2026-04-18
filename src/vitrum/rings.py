from vitrum.glass_Atoms import GlassAtoms
import numpy as np
from scipy.sparse import csr_array, coo_matrix
from scipy.sparse.csgraph import dijkstra
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from ase.symbols import symbols2numbers
from ase.io import write
from ase import Atoms
from collections import Counter
from typing import List, Dict, Optional, Tuple, Set, Union


def check_ring_is_periodic(ring: List[int], offsets: Dict[Tuple[int, int], np.ndarray]) -> bool:
    ''' 
    Check if the ring wraps around the period cell, i.e., is not a true ring.
    
    Args:
        ring (List[int]): Atom indices of the ring.
        offsets (Dict[Tuple[int, int], np.ndarray]): Unit cell offsets for all atoms pairs.

    Returns:
        bool: True if the ring is periodic (wraps around), False otherwise (contained within cell without wrapping sum).
    '''
    total_offset = np.zeros(3)
    for i in range(len(ring) - 1):
        total_offset += offsets[(ring[i], ring[i+1])]
    total_offset += offsets[(ring[-1], ring[0])]
    return np.all(total_offset == 0)


def find_rings(
    ats: Atoms,
    radii_factor: float = 1.3,
    repeat: Tuple[int, int, int] = (1, 1, 1),
    bonds: Optional[List[Tuple[str, str]]] = None,
    limit: float = np.inf
) -> List[List[int]]:
    ''' 
    Find rings in the unit cell.
    Rings are found according to the definition of L. Guttman, J. Non-Cryst. Solids 1990, 116.
    
    Args:
        ats (ase.Atoms): Atoms object containing the structure
        radii_factor (float): Factor to multiply covalent radii for neighbor search
        repeat (Tuple[int, int, int]): How often to repeat the unit cell in each direction. Increase for small cells.
        bonds (Optional[List[Tuple[str, str]]]): List of allowed bonds, e.g., [('C', 'C'), ('C', 'O')], can be None to allow all bonds.
        limit (float): Maximum ring size to search for.

    Returns:
        List[List[int]]: A list of rings, where each ring is a list of atom indices.
    '''
    s = ats.repeat(repeat)
    pos = s.get_positions()
    nat = len(s)
    lat = s.get_cell()
    els = s.get_chemical_symbols()
    radii = covalent_radii[symbols2numbers(els)]

    if bonds is not None:
        # Don't need to find neighbors for elements not included in bonds
        elements = set().union(*bonds)
        radii = [x if el in elements else 0. for el, x in zip(els, radii)]
        radii = np.array(radii, dtype=float)
           
    nl = NeighborList(radii * radii_factor, self_interaction=False, bothways=False, skin=0.)
    nl.update(s)

    d_idx = []
    d_val = []
    # unit cell offsets for all atoms
    all_offsets = {}

    for i in range(nat):
        indices, offsets = nl.get_neighbors(i)

        rs = pos[indices, :] + offsets @ lat - pos[i, :]
        ds = np.linalg.norm(rs, axis=1)
        for j, r, o in zip(indices, ds, offsets):
            # Ignore bonds that are not included
            if ((els[i], els[j]) in bonds or (els[j], els[i]) in bonds):
                d_idx.append((i, j))
                d_val.append(r)
                d_idx.append((j, i))
                d_val.append(r)
                all_offsets[(i, j)] = o
                all_offsets[(j, i)] = -o


    # sparse matrix of bonds, removes zero entries
    d = csr_array((d_val, np.array(d_idx, dtype=np.int32).T), shape=(nat, nat))

    # now find the rings 
    rings = {}
    for i in range(len(ats)):
        indices, offsets = nl.get_neighbors(i)
        for j, _ in zip(indices, offsets):
            if bonds is not None: 
                # skip if bond is not allowed
                if not ((els[i], els[j]) in bonds or (els[j], els[i]) in bonds):
                    continue
            # Remove the bond between the two selected neighbors
            d_tmp = d.copy()
            d_tmp[i, j] = 0
            d_tmp[j, i] = 0
            d_tmp.eliminate_zeros()
            # Now find the shortest path between i and j
            dist_matrix, predecessors = dijkstra(d_tmp, indices=i, return_predecessors=True, directed=False, unweighted=True, limit=limit)
            if dist_matrix[j] < np.inf:
                k = j
                ring = [k]
                while predecessors[k] != i:
                    k = predecessors[k]
                    ring.append(k)
                ring.append(i)

                if not check_ring_is_periodic(ring, all_offsets):
                    print('WARNING: ring is wrapping around periodic cell! Consider increasing `repeat`.')
                    continue

                ring = [x % len(ats) for x in ring]  # take it back to primary cell
                rings[tuple(sorted(ring))] = ring


    return list(rings.values())

class Ring(object):
    """
    A class representing a ring in a atomistic system.
    """

    def __init__(self, atoms: Atoms, indexes: Optional[List[int]] = None):
        """
        Initialize a Ring object.

        Args:
            atoms (Atoms): An Atoms object representing the atoms in the ring.
            indexes (Optional[List[int]], optional): A list of indices of the atoms involved in the ring. Defaults to None.
        """
        self.atoms = atoms
        self.indexes = indexes
        self._roundness = None
        self._roughness = None
        self.ellipsoid_lengths = None
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = [np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types]

    def center(self) -> np.ndarray:
        """
        Calculate the center of the ring.

        Returns:
            np.ndarray: An array of shape (3,) representing the center of the ring.
        """
        positions = self.atoms.get_positions()
        cell = np.diagonal(self.atoms.get_cell())
        differences = np.diff(positions, axis=0, append=positions[:1])
        crossings = np.floor_divide(differences + 0.5 * cell, cell).astype(int)
        adjusted_positions = positions - np.cumsum(cell * crossings, axis=0)
        center = np.mean(adjusted_positions, axis=0)
        center = np.mod(center + cell, cell)
        return center

    def size(self) -> int:
        """
        Calculate the size of the ring, i.e., the number of atoms in the ring.

        Returns:
            int: The size of the ring.
        """
        return len(self.atoms)


class RingAnalysis:
    """
    A class for calculating and analyzing rings in atomistic systems.
    """

    def __init__(self, atoms: Atoms, included_atoms: List[str], bonding_dict: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize the RingAnalysis class.

        Args:
            atoms (Atoms): An Atoms object representing the atoms in the system.
            included_atoms (List[str]): A list of strings representing the chemical symbols of the atoms to include in the analysis.
            bonding_dict (Optional[List[Tuple[str, str]]]): A list of allowed bonds, e.g., [('Si', 'O')].
        """
        super().__init__()
        self.bonding_dict = bonding_dict
        atoms = atoms[[atom.symbol in included_atoms for atom in atoms]]
        self.atoms = GlassAtoms(atoms)
        self.num_atoms = len(self.atoms)
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = [np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types]
        self.rings = None


    def calculate(
        self,
        radii_factor: float = 1.3,
        repeat: Tuple[int, int, int] = (1, 1, 1),
        max_size: float = np.inf
    ) -> List[Ring]:
        """
        Calculate the rings in the system.

        Args:
            radii_factor (float): Factor to multiply covalent radii for neighbor search.
            repeat (Tuple[int, int, int]): Repeat unit cell.
            max_size (float): Maximum ring size.

        Returns:
            List[Ring]: A list of Ring objects representing the rings in the system.
        """
        bonds = self.bonding_dict

        rings = find_rings(ats=self.atoms, radii_factor=radii_factor, repeat=repeat, bonds=bonds, limit=max_size)

        self.rings = [Ring(self.atoms[list(r)], list(r)) for r in rings]
        return self.rings

    def write_rings(self, filename: str, format: str = 'extxyz'):
        """
        Write the rings to a file.

        Args:
            filename (str): The name of the file to write the rings to.
            format (str): The format of the file.
        """
        if self.rings is None:
            raise ValueError("Rings have not been calculated yet.")
        write(filename, [r.atoms for r in self.rings], format=format)
    
    def get_ring_size_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of ring sizes.

        Returns:
            Dict[int, int]: A dictionary where keys are ring sizes and values are counts.
        """
        if self.rings is None:
            raise ValueError("Rings have not been calculated yet.")
        ring_sizes = [len(r.atoms) for r in self.rings]
        return dict(Counter(ring_sizes))
    
    def plot_ring_size_distribution(self, ax=None):
            """
            Plots the distribution of ring sizes using matplotlib.
            """
            import matplotlib.pyplot as plt
            
            dist = self.get_ring_size_distribution()
            if not dist:
                print("No rings found. Ensure you have run .calculate() first.")
                return

            sizes = sorted(dist.keys())
            counts = np.array([dist[size] for size in sizes])
            frequency = counts / np.sum(counts)

            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.set_xlabel('Ring Size', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_xticks(sizes)  

            ax.plot(sizes, frequency)

            return ax

# Alias for backward compatibility
RINGs = RingAnalysis