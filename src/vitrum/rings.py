from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import write
from ase.neighborlist import NeighborList
from ase.symbols import symbols2numbers
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from vitrum.glass_atoms import GlassAtoms


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


def _walk_path(predecessors: np.ndarray, source: int, target: int) -> List[int]:
    '''
    Reconstruct a dijkstra shortest path from `source` to `target`.

    Args:
        predecessors (np.ndarray): Predecessor array returned by `dijkstra` for the search rooted at `source`.
        source (int): The atom index the dijkstra search was rooted at.
        target (int): The atom index to reconstruct the path to.

    Returns:
        List[int]: The path as [target, ..., source].
    '''
    k = target
    path = [k]
    while predecessors[k] != source:
        k = predecessors[k]
        path.append(k)
    path.append(source)
    return path


def _find_guttman_rings(
    len_ats: int,
    d: csr_array,
    limit: float,
) -> List[List[int]]:
    '''
    Find raw Guttman ring candidates (before periodicity check and remapping to the primary cell).

    For each bond (i, j), remove that edge and find the shortest path between i and j in the
    remaining graph. The path plus the removed edge forms the ring.
    '''
    raw_rings = []
    for i in range(len_ats):
        neighbors = d.indices[d.indptr[i]:d.indptr[i + 1]]
        for j in neighbors:
            d_tmp = d.copy()
            d_tmp[i, j] = 0
            d_tmp[j, i] = 0
            d_tmp.eliminate_zeros()
            dist_matrix, predecessors = dijkstra(
                d_tmp, indices=i, return_predecessors=True, directed=False, unweighted=True, limit=limit
            )
            if dist_matrix[j] < np.inf:
                raw_rings.append(_walk_path(predecessors, i, j))
    return raw_rings


def _find_king_candidate_rings(
    len_ats: int,
    d: csr_array,
    limit: float,
) -> List[List[int]]:
    '''
    Find raw King's ring candidates (before periodicity check and remapping to the primary cell).

    For each atom c and each pair of its bonded neighbors (n1, n2), remove c from the graph and
    find the shortest path between n1 and n2 in the remaining graph. The path plus the two edges
    to c forms the ring.
    '''
    raw_rings = []
    for c in range(len_ats):
        neighbors = sorted(d.indices[d.indptr[c]:d.indptr[c + 1]])
        if len(neighbors) < 2:
            continue

        d_tmp = d.copy()
        d_tmp[c, :] = 0
        d_tmp[:, c] = 0
        d_tmp.eliminate_zeros()

        # One multi-source search from all of c's neighbors covers every pair at once,
        # instead of re-running a single-source search per pair.
        dist_matrix, predecessors = dijkstra(
            d_tmp, indices=neighbors, return_predecessors=True, directed=False, unweighted=True, limit=limit
        )
        for p, n1 in enumerate(neighbors):
            for n2 in neighbors[p + 1:]:
                if dist_matrix[p, n2] < np.inf:
                    raw_rings.append(_walk_path(predecessors[p], n1, n2) + [c])
    return raw_rings


def _find_primitive_rings(
    len_ats: int,
    d: csr_array,
    limit: float,
) -> List[List[int]]:
    '''
    Find raw primitive rings (before periodicity check and remapping to the primary cell).

    Generates King's ring candidates and keeps only those that cannot be decomposed into two
    smaller rings: for every pair of atoms in the ring, the shortest path between them in the
    full bond graph must equal the geodesic distance between them measured along the ring. Any
    pair connected by a shorter path (a "shortcut") means the ring decomposes and is rejected.
    '''
    candidates = _find_king_candidate_rings(len_ats, d, limit)
    primitive_rings = []
    for ring in candidates:
        n = len(ring)
        # A shortcut can only matter if it's shorter than the longest possible ring-arc
        # distance (n // 2), so the search never needs to look further than that.
        dist_matrix = dijkstra(d, indices=ring, directed=False, unweighted=True, limit=n // 2)
        is_primitive = True
        for p in range(n):
            for q in range(p + 1, n):
                ring_dist = min(q - p, n - (q - p))
                if dist_matrix[p, ring[q]] < ring_dist:
                    is_primitive = False
                    break
            if not is_primitive:
                break
        if is_primitive:
            primitive_rings.append(ring)
    return primitive_rings


def find_rings(
    ats: Atoms,
    radii_factor: float = 1.3,
    repeat: Tuple[int, int, int] = (1, 1, 1),
    bonds: Optional[List[Tuple[str, str]]] = None,
    limit: float = np.inf,
    criterion: str = "guttman",
) -> List[List[int]]:
    '''
    Find rings in the unit cell.

    Three ring criteria are supported via `criterion`:

    - "guttman": L. Guttman, J. Non-Cryst. Solids 1990, 116. For each bond (i, j), remove that
      edge and find the shortest path between i and j in the remaining graph.
    - "king": S. V. King, Nature 1967, 213, 425. For each atom c and each pair of its bonded
      neighbors (n1, n2), remove c from the graph and find the shortest path between n1 and n2
      in the remaining graph.
    - "primitive": D. S. Franzblau, Phys. Rev. B 1991, 44, 4925. A ring that cannot be
      decomposed into two smaller rings. Found by generating King's ring candidates and
      rejecting any with a "shortcut": a pair of ring atoms connected, in the full bond graph,
      by a path shorter than the corresponding arc of the ring.

    Args:
        ats (ase.Atoms): Atoms object containing the structure
        radii_factor (float): Factor to multiply covalent radii for neighbor search
        repeat (Tuple[int, int, int]): How often to repeat the unit cell in each direction. Increase for small cells.
        bonds (Optional[List[Tuple[str, str]]]): List of allowed bonds, e.g., [('C', 'C'), ('C', 'O')], can be None to allow all bonds.
        limit (float): Maximum ring size to search for.
        criterion (str): Ring criterion to use, one of "guttman", "king", or "primitive".

    Returns:
        List[List[int]]: A list of rings, where each ring is a list of atom indices.
    '''
    if criterion not in ("guttman", "king", "primitive"):
        raise ValueError(f"Unknown ring criterion '{criterion}', expected 'guttman', 'king', or 'primitive'.")

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

    # now find the rings, according to the requested criterion
    if criterion == "guttman":
        raw_rings = _find_guttman_rings(len(ats), d, limit)
    elif criterion == "king":
        raw_rings = _find_king_candidate_rings(len(ats), d, limit)
    else:
        raw_rings = _find_primitive_rings(len(ats), d, limit)

    rings = {}
    for ring in raw_rings:
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
        self._unwrapped_positions_cache = None
        self._roundness = None
        self._roughness = None
        self.ellipsoid_lengths = None
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = {atom_type: np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types}

    def _unwrapped_positions(self) -> np.ndarray:
        """
        Ring atom positions with periodic-boundary jumps removed.

        Rings are only ever kept by `find_rings` if they don't wind around the periodic
        cell (see `check_ring_is_periodic`), so every atom has a single well-defined
        position relative to the first ring atom: its minimum-image displacement from it.

        Returns:
            np.ndarray: An array of shape (size(), 3) of PBC-unwrapped Cartesian positions.
        """
        if self._unwrapped_positions_cache is None:
            positions = self.atoms.get_positions()
            displacements, _ = find_mic(positions - positions[0], self.atoms.get_cell(), self.atoms.get_pbc())
            self._unwrapped_positions_cache = positions[0] + displacements
        return self._unwrapped_positions_cache

    def center(self) -> np.ndarray:
        """
        Calculate the center of the ring.

        Returns:
            np.ndarray: An array of shape (3,) representing the center of the ring, wrapped into the primary cell.
        """
        center = self._unwrapped_positions().mean(axis=0)
        cell = self.atoms.get_cell()
        return cell.cartesian_positions(cell.scaled_positions(center) % 1.0)

    def size(self) -> int:
        """
        Calculate the size of the ring, i.e., the number of atoms in the ring.

        Returns:
            int: The size of the ring.
        """
        return len(self.atoms)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the ring, i.e., the sum of the consecutive bond
        lengths around the ring (including the closing bond from the last atom back
        to the first).

        Returns:
            float: The perimeter of the ring.
        """
        xyz = self._unwrapped_positions()
        edges = np.diff(xyz, axis=0, append=xyz[:1])
        return float(np.linalg.norm(edges, axis=1).sum())

    def _compute_ellipsoid(self) -> None:
        """
        Fit the ring atoms with a best-fit ellipsoid via the SVD of their centered,
        PBC-unwrapped positions. The resulting singular values (descending) are the
        lengths of the ellipsoid's three principal axes and are the basis for
        `roundness`, `roughness`, and `radius_of_gyration`.
        """
        xyz = self._unwrapped_positions()
        xyz = xyz - xyz.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(xyz)
        self.ellipsoid_lengths = singular_values

    def roundness(self) -> float:
        """
        Calculate the roundness of the ring: the ratio of the second-largest to the
        largest best-fit ellipsoid axis, i.e. how circular (close to 1) vs. elongated
        (close to 0) the ring is within its own best-fit plane.

        Returns:
            float: The roundness of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        if self._roundness is None:
            self._roundness = self.ellipsoid_lengths[1] / self.ellipsoid_lengths[0]
        return self._roundness

    def roughness(self) -> float:
        """
        Calculate the roughness of the ring: the ratio of the smallest best-fit
        ellipsoid axis to the geometric mean of the other two, i.e. how far the ring
        deviates out of its own best-fit plane.

        Returns:
            float: The roughness of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        if self._roughness is None:
            self._roughness = self.ellipsoid_lengths[2] / np.sqrt(self.ellipsoid_lengths[0] * self.ellipsoid_lengths[1])
        return self._roughness

    def radius_of_gyration(self) -> float:
        """
        Calculate the radius of gyration of the ring: the root-mean-square distance
        of the ring atoms from their centroid.

        Returns:
            float: The radius of gyration of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        return float(np.sqrt(np.sum(self.ellipsoid_lengths ** 2) / self.size()))


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
        max_size: float = np.inf,
        criterion: str = "guttman",
    ) -> List[Ring]:
        """
        Calculate the rings in the system.

        Args:
            radii_factor (float): Factor to multiply covalent radii for neighbor search.
            repeat (Tuple[int, int, int]): Repeat unit cell.
            max_size (float): Maximum ring size.
            criterion (str): Ring criterion to use, one of "guttman", "king", or "primitive".
                See `vitrum.rings.find_rings` for details on each criterion.

        Returns:
            List[Ring]: A list of Ring objects representing the rings in the system.
        """
        bonds = self.bonding_dict

        rings = find_rings(
            ats=self.atoms, radii_factor=radii_factor, repeat=repeat, bonds=bonds, limit=max_size, criterion=criterion
        )

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
    
    def plot_ring_size_distribution(self, ax=None, **plot_kwargs):
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
            frequency = counts / self.atoms.get_volume()  # Normalize by volume to get frequency

            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.set_xlabel('Ring Size (N$_{atoms}$)', fontsize=12)
                ax.set_ylabel('Ring Frequency [N$_{rings}$ / V] (Å$^{-3}$)', fontsize=12)
                ax.set_xticks(sizes)  

            ax.plot(sizes, frequency, **plot_kwargs)

            return ax