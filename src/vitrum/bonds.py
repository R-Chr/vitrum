"""Bond adjacency between two selections of atoms.

Coordination numbers, neighbour lists, bond angles and Q^n speciation are all reductions of
one object: which atoms of a centre selection lie within a cutoff of which atoms of a
neighbour selection. `Bonds` is that object, held as an edge list rather than an N x N
matrix.
"""

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from vitrum.geometry import pdf, radial_bins


class Bonds:
    """
    Which atoms of one selection are bonded to which atoms of another.

    Bond ``k`` joins ``centers[row[k]]`` to ``neighs[col[k]]``. A bond is a pair whose
    minimum-image separation is below the cutoff; an atom is never bonded to itself, though
    it may appear in both selections.

    Attributes:
        centers (np.ndarray): Global atom indices of the centre selection.
        neighs (np.ndarray): Global atom indices of the neighbour selection.
        row (np.ndarray): Position in `centers` of each bond's centre atom.
        col (np.ndarray): Position in `neighs` of each bond's neighbour atom.
        offsets (np.ndarray): `(len(row), 3)` integer cell-boundary crossings, one row per
            bond, such that ``D = pos[neighs[col]] - pos[centers[row]] + offsets @ cell`` is
            the bond's minimum-image displacement vector.
    """

    __slots__ = ("centers", "col", "neighs", "offsets", "row")

    def __init__(self, centers, neighs, row, col, offsets=None):
        self.centers = np.asarray(centers, dtype=int)
        self.neighs = np.asarray(neighs, dtype=int)
        self.row = np.asarray(row, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self.offsets = (
            np.zeros((len(self.row), 3), dtype=int)
            if offsets is None
            else np.asarray(offsets, dtype=int)
        )

    def __len__(self) -> int:
        return int(self.row.size)

    def __repr__(self) -> str:
        return (
            f"Bonds({len(self.centers)} centers, {len(self.neighs)} neighbours, "
            f"{len(self)} bonds)"
        )

    def counts(self) -> np.ndarray:
        """Bonds at each centre atom, i.e. its coordination number."""
        return np.bincount(self.row, minlength=len(self.centers))

    def degrees(self) -> np.ndarray:
        """Bonds at each neighbour atom. A bridging atom is one with two or more."""
        return np.bincount(self.col, minlength=len(self.neighs))

    def neighbors_of(self, i: int) -> np.ndarray:
        """Sorted global indices bonded to centre `i`, a position in `centers`."""
        return np.sort(self.neighs[self.col[self.row == i]])

    def lists(self) -> list[np.ndarray]:
        """`neighbors_of` every centre atom, in the order of `centers`."""
        ordered = self.neighs[self.col[np.argsort(self.row, kind="stable")]]
        splits = np.cumsum(self.counts())[:-1]
        return [np.sort(part) for part in np.split(ordered, splits)]

    def select_neighs(self, keep: np.ndarray) -> "Bonds":
        """Drop every bond whose neighbour is False in `keep`, one entry per `neighs` atom."""
        edges = np.asarray(keep, dtype=bool)[self.col]
        return Bonds(
            self.centers, self.neighs, self.row[edges], self.col[edges], self.offsets[edges]
        )

    def matrix(self) -> np.ndarray:
        """The adjacency as a dense (len(centers), len(neighs)) boolean matrix."""
        out = np.zeros((len(self.centers), len(self.neighs)), dtype=bool)
        out[self.row, self.col] = True
        return out


def _as_list(symbols: str | Sequence[str]) -> list[str]:
    """Read a bare chemical symbol as a one-species selection."""
    return [symbols] if isinstance(symbols, str) else list(symbols)


class _Frame:
    """One frame's species lookup, backing the bond and PDF queries made against it.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.types = np.array(atoms.get_chemical_symbols())
        self.species = np.unique(self.types)

    def require(self, *symbols: str) -> None:
        """Raise a ValueError naming any of `symbols` absent from the frame."""
        present = set(self.species.tolist())
        missing = [s for s in symbols if s not in present]
        if missing:
            raise ValueError(
                f"Species {missing} not present in the structure. "
                f"Available species: {sorted(present)}."
            )

    def index(self, *symbols: str) -> np.ndarray:
        """Sorted global indices of every atom of the given species, each listed once."""
        return np.flatnonzero(np.isin(self.types, symbols))

    def bonds(
        self,
        center_types: str | Sequence[str],
        neigh_types: str | Sequence[str],
        cutoff: float,
    ) -> Bonds:
        """Every bond within `cutoff` between two species selections."""
        centers = self.index(*_as_list(center_types))
        neighs = self.index(*_as_list(neigh_types))
        return _neighbor_list_bonds(self.atoms, centers, neighs, cutoff)

    def partial_pdf(
        self, pair: tuple[str, str], rrange: float, nbin: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """g_ab(r) as `geometry.partial_pdf` defines it, from a neighbor list."""
        volume = self.atoms.get_volume()
        first, second = self.index(pair[0]), self.index(pair[1])
        if len(first) == 0 or len(second) == 0:
            return radial_bins(rrange, nbin)[0], np.zeros(nbin)

        # Match the dense block's pair counting: a like pair contributes each bond twice,
        # once per ordering. Its zero diagonal has no counterpart here.
        like_pair = pair[0] == pair[1]
        n_pairs = len(first) * (len(first) - 1) if like_pair else len(first) * len(second)
        i, j, d, _ = _min_image_edges(self.atoms, rrange)
        row = _positions_in(first, i)
        col = _positions_in(second, j)
        keep = (row >= 0) & (col >= 0)
        return pdf(d[keep], volume, rrange, nbin, n_pairs=n_pairs, exclude_self=False)


def _min_image_edges(
    atoms: Atoms, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every directed pair within `cutoff`, deduped to one row per minimum-image bond.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Global indices `i`, `j`, the
            bond distance `d`, and the shift vector `S` (``D = pos[j]-pos[i]+S@cell``), one
            row per kept directed pair.
    """
    empty = np.empty(0, dtype=int)
    empty_offsets = np.empty((0, 3), dtype=int)
    i, j, d, S = neighbor_list("ijdS", atoms, cutoff)
    keep = i != j  # an atom is never bonded to itself, including its own periodic image
    i, j, d, S = i[keep], j[keep], d[keep], S[keep]
    if i.size == 0:
        return empty, empty, np.empty(0), empty_offsets

    # Sort by centre, then neighbour, then distance, so the first row of each (i, j) run is
    # its closest image.
    order = np.lexsort((d, j, i))
    i, j, d, S = i[order], j[order], d[order], S[order]
    first = np.empty(len(i), dtype=bool)
    first[0] = True
    first[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1])
    return i[first], j[first], d[first], S[first]


def _neighbor_list_bonds(
    atoms: Atoms, centers: np.ndarray, neighs: np.ndarray, cutoff: float
) -> Bonds:
    """Bonds from `ase.neighborlist.neighbor_list`, costing the bond count rather than N^2."""
    empty = np.empty(0, dtype=int)
    empty_offsets = np.empty((0, 3), dtype=int)
    if len(centers) == 0 or len(neighs) == 0:
        return Bonds(centers, neighs, empty, empty, empty_offsets)

    i, j, _, S = _min_image_edges(atoms, cutoff)
    row = _positions_in(centers, i)
    col = _positions_in(neighs, j)
    keep = (row >= 0) & (col >= 0)
    return Bonds(centers, neighs, row[keep], col[keep], S[keep])


def graph_edges(
    atoms: Atoms, pair_cutoffs: dict[tuple[str, str], float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Every directed bond in `atoms` under a per-species-pair cutoff, every periodic image kept.

    Args:
        atoms (Atoms): The structure to search, generally a repeated supercell.
        pair_cutoffs (Dict[Tuple[str, str], float]): Cutoff for each species pair to search;
            a pair absent from the dict is not searched. Key order does not matter.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Global atom indices `i`, `j`,
            the bond distance `d`, and the shift vector `S` such that
            ``D = pos[j] - pos[i] + S @ cell``, one row per directed edge.
    """
    cutoff = {key: float(value) for key, value in pair_cutoffs.items()}
    return neighbor_list("ijdS", atoms, cutoff)


def _positions_in(selection: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position of each value within the sorted `selection`, or -1 where it is absent."""
    where = np.clip(np.searchsorted(selection, values), 0, len(selection) - 1)
    return np.where(selection[where] == values, where, -1)
