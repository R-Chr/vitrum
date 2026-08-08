"""Tests for vitrum.bonds, the adjacency primitive under Coordination.

`Bonds` reductions (`counts`, `degrees`, `lists`, `matrix`, ...) are checked for internal
consistency, and the `neighbor_list`-based bond finder is checked against independent
references: brute-force minimum images, and `ase.geometry.find_mic`.
"""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic
from conftest import (
    CAF2_FIRST_SHELL_CUTOFF,
    SI_FIRST_SHELL_CUTOFF,
    SI_NN_DISTANCE,
    SI_O_CUTOFF,
)

from vitrum.bonds import Bonds, _Frame, _neighbor_list_bonds


def _kd_bonds(atoms, center_type, neigh_type, cutoff):
    """Bonds via the neighbor list, bypassing the frame's species lookup for convenience."""
    frame = _Frame(atoms)
    centers = frame.index(*([center_type] if isinstance(center_type, str) else center_type))
    neighs = frame.index(*([neigh_type] if isinstance(neigh_type, str) else neigh_type))
    return _neighbor_list_bonds(atoms, centers, neighs, cutoff)


def _same(a: Bonds, b: Bonds) -> bool:
    return (
        np.array_equal(a.centers, b.centers)
        and np.array_equal(a.neighs, b.neighs)
        and np.array_equal(a.counts(), b.counts())
        and np.array_equal(a.degrees(), b.degrees())
        and all(np.array_equal(x, y) for x, y in zip(a.lists(), b.lists()))
    )


def _brute_force_counts(atoms, first: np.ndarray, second: np.ndarray, cutoff: float) -> np.ndarray:
    """Reference coordination number of each `first` atom, via `find_mic` rather than `Bonds`.

    `Bonds` counts one minimum-image bond per pair regardless of how the cutoff compares to
    the cell size, and `find_mic` gives exactly that image, so this reference is valid past
    half the cell length too.
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    counts = np.empty(len(first), dtype=int)
    for k, i in enumerate(first):
        _, lengths = find_mic(positions[second] - positions[i], cell, pbc=True)
        distinct = second != i
        counts[k] = np.count_nonzero((lengths < cutoff) & distinct)
    return counts


@pytest.mark.parametrize("cutoff", [1.5, 2.5, 3.0, 4.2, 6.0])
def test_neighbor_list_matches_brute_force_on_silicon(silicon_small, cutoff):
    """silicon_small is a 10.86 A cell, so cutoffs of 6.0 exceed L/2 and exercise the case
    where a pair has more than one periodic image inside the cutoff.
    """
    bonds = _kd_bonds(silicon_small, "Si", "Si", cutoff)
    np.testing.assert_array_equal(
        bonds.counts(), _brute_force_counts(silicon_small, bonds.centers, bonds.neighs, cutoff)
    )


@pytest.mark.parametrize(
    "center, neigh", [("Ca", "F"), ("F", "Ca"), ("F", "F"), (["Ca", "F"], "F")]
)
def test_neighbor_list_matches_brute_force_on_fluorite(fluorite_caf2, center, neigh):
    """Cross pairs, like pairs and multi-species selections must all match the reference."""
    bonds = _kd_bonds(fluorite_caf2, center, neigh, CAF2_FIRST_SHELL_CUTOFF)
    np.testing.assert_array_equal(
        bonds.counts(),
        _brute_force_counts(fluorite_caf2, bonds.centers, bonds.neighs, CAF2_FIRST_SHELL_CUTOFF),
    )


def test_neighbor_list_matches_brute_force_on_a_disordered_structure(sodium_silicate_frame):
    """The same reference on a real melt: 3000 atoms, no symmetry to hide a mistake behind."""
    bonds = _kd_bonds(sodium_silicate_frame, "Si", "O", SI_O_CUTOFF)
    np.testing.assert_array_equal(
        bonds.counts(),
        _brute_force_counts(sodium_silicate_frame, bonds.centers, bonds.neighs, SI_O_CUTOFF),
    )


def test_bond_finding_is_independent_of_wrapping(silicon_small):
    """Bond finding may not depend on the positions having been wrapped into the cell."""
    shifted = silicon_small.copy()
    shifted.set_positions(shifted.get_positions() + np.array([53.0, -71.0, 17.0]))
    assert _same(
        _kd_bonds(silicon_small, "Si", "Si", SI_FIRST_SHELL_CUTOFF),
        _kd_bonds(shifted, "Si", "Si", SI_FIRST_SHELL_CUTOFF),
    )


def test_neighbor_list_matches_brute_force_on_a_triclinic_cell(triclinic_atoms):
    """An independent minimum image, by brute force over the 27 nearest cell images.

    The cutoff is kept well under half the shortest cell vector, so the 27-image search is
    exact.
    """
    cutoff = SI_FIRST_SHELL_CUTOFF
    positions = triclinic_atoms.get_positions()
    cell = np.array(triclinic_atoms.get_cell())
    n = len(triclinic_atoms)

    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ cell
    images = positions[None, :, :] + shifts[:, None, :]  # (27, n, 3)
    counts = np.empty(n, dtype=int)
    for i in range(n):
        dist = np.linalg.norm(images - positions[i], axis=2).min(axis=0)
        dist[i] = np.inf  # exclude the atom itself
        counts[i] = np.count_nonzero(dist < cutoff)

    bonds = _kd_bonds(triclinic_atoms, "Si", "Si", cutoff)
    np.testing.assert_array_equal(bonds.counts(), counts)


def test_offsets_reproduce_the_bond_distance(triclinic_atoms):
    """D = pos[j] - pos[i] + offsets @ cell must reproduce the minimum-image bond vector.

    Checked against `ase.geometry.find_mic`, a minimum-image implementation independent of
    `neighbor_list` itself.
    """
    bonds = _kd_bonds(triclinic_atoms, "Si", "Si", SI_FIRST_SHELL_CUTOFF)
    positions = triclinic_atoms.get_positions()
    cell = triclinic_atoms.get_cell()

    raw = positions[bonds.neighs[bonds.col]] - positions[bonds.centers[bonds.row]]
    reconstructed = raw + bonds.offsets @ np.array(cell)
    expected, _ = find_mic(raw, cell, pbc=True)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-8)


def test_exact_cutoff_distance_is_excluded():
    """`neighbor_list`, and so `Bonds`, is documented strict `< cutoff`."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[10.0] * 3, pbc=True)
    assert len(_kd_bonds(atoms, "H", "H", 2.0)) == 0
    # A cutoff a hair past it must bond, confirming the exclusion is the cutoff and not a
    # broken setup.
    assert len(_kd_bonds(atoms, "H", "H", 2.0 + 1e-9)) == 2


def test_coincident_distinct_atoms_bond():
    """Two distinct atoms at the same position are still a bonded pair, not a self-pair."""
    atoms = Atoms(
        "H2", positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], cell=[10.0] * 3, pbc=True
    )
    assert _kd_bonds(atoms, "H", "H", 1.0).counts().tolist() == [1, 1]


def test_a_cluster_with_no_cell_still_bonds():
    """neighbor_list handles a cluster directly; no periodic box is needed."""
    cluster = Atoms("H3", positions=[[0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    bonds = _Frame(cluster).bonds("H", "H", 2.0)
    assert bonds.counts().tolist() == [1, 1, 0]


def test_counts_and_degrees_are_the_two_margins(silicon_small):
    bonds = _Frame(silicon_small).bonds("Si", "Si", SI_FIRST_SHELL_CUTOFF)
    assert bonds.counts().sum() == len(bonds)
    assert bonds.degrees().sum() == len(bonds)
    assert set(bonds.counts().tolist()) == {4}


def test_lists_returns_global_indices(fluorite_caf2):
    bonds = _Frame(fluorite_caf2).bonds("Ca", "F", CAF2_FIRST_SHELL_CUTOFF)
    symbols = np.array(fluorite_caf2.get_chemical_symbols())
    for entry in bonds.lists():
        assert set(symbols[entry].tolist()) == {"F"}


def test_select_neighs_drops_only_the_failing_neighbours(fluorite_caf2):
    bonds = _Frame(fluorite_caf2).bonds("Ca", "F", CAF2_FIRST_SHELL_CUTOFF)
    keep = np.zeros(len(bonds.neighs), dtype=bool)
    keep[: len(keep) // 2] = True
    kept = bonds.select_neighs(keep)
    for before, after in zip(bonds.lists(), kept.lists()):
        # `neighs` is sorted and every list is a sorted subset of it, so searchsorted
        # maps each global index back to its flag.
        np.testing.assert_array_equal(after, before[keep[np.searchsorted(bonds.neighs, before)]])
    assert len(kept) < len(bonds)


def test_an_atom_is_never_bonded_to_itself(silicon_small):
    """The two selections overlap for a like pair, so the diagonal has to be excluded."""
    bonds = _Frame(silicon_small).bonds("Si", "Si", SI_FIRST_SHELL_CUTOFF)
    for centre, entry in zip(bonds.centers, bonds.lists()):
        assert centre not in entry


def test_absent_species_gives_an_empty_selection(silicon_small):
    bonds = _Frame(silicon_small).bonds("Si", "Ge", SI_FIRST_SHELL_CUTOFF)
    assert len(bonds) == 0
    assert bonds.counts().tolist() == [0] * len(silicon_small)


def test_index_deduplicates_a_repeated_species(silicon_small):
    """A species named twice must not contribute its atoms twice.

    get_bridging_analysis(former_types=["Si", "Si"]) would otherwise double every former's
    contribution, which can make the two-formers test for a bridging atom pass spuriously.
    """
    frame = _Frame(silicon_small)
    once = frame.index("Si")
    twice = frame.index("Si", "Si")
    np.testing.assert_array_equal(once, twice)


def test_frame_reports_absent_species(silicon_small):
    with pytest.raises(ValueError, match="not present in the structure"):
        _Frame(silicon_small).require("Ge")


def test_lengths_recover_the_silicon_bond_length(silicon_small):
    """Every first-shell bond in diamond silicon is a*sqrt(3)/4 long, PBC or not."""
    bonds = _Frame(silicon_small).bonds("Si", "Si", SI_FIRST_SHELL_CUTOFF)
    lengths = bonds.lengths(silicon_small)
    assert len(lengths) == len(bonds)
    np.testing.assert_allclose(lengths, SI_NN_DISTANCE, rtol=1e-10)


@pytest.mark.parametrize("cutoff", [2.5, 4.2, 6.0])
def test_lengths_match_find_mic(silicon_small, cutoff):
    """The offsets must name the same image `find_mic` picks, at any cutoff."""
    bonds = _Frame(silicon_small).bonds("Si", "Si", cutoff)
    positions = silicon_small.get_positions()
    expected = find_mic(
        positions[bonds.neighs[bonds.col]] - positions[bonds.centers[bonds.row]],
        silicon_small.get_cell(),
        pbc=True,
    )[1]
    np.testing.assert_allclose(bonds.lengths(silicon_small), expected, atol=1e-10)


def test_lengths_of_no_bonds_is_empty(silicon_small):
    bonds = _Frame(silicon_small).bonds("Si", "Si", 0.5)
    assert bonds.lengths(silicon_small).shape == (0,)
