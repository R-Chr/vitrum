"""Tests for vitrum.glass_atoms and the orthorhombic-cell guard."""

import numpy as np
import pytest

from vitrum.geometry import require_orthorhombic
from vitrum.glass_atoms import GlassAtoms

SI_FIRST_SHELL_CUTOFF = 3.0


def test_get_dist_matches_ase_minimum_image(silicon_small):
    """The numba distance kernel must agree with ASE's own MIC distances."""
    atoms = GlassAtoms(silicon_small)
    np.testing.assert_allclose(
        atoms.get_dist(), atoms.get_all_distances(mic=True), atol=1e-8
    )


def test_get_dist_is_symmetric(silicon_small):
    distances = GlassAtoms(silicon_small).get_dist()
    np.testing.assert_allclose(distances, distances.T, atol=1e-12)


def test_get_dist_rejects_triclinic(triclinic_atoms):
    """A triclinic cell must raise rather than silently give wrong distances."""
    atoms = GlassAtoms(triclinic_atoms)
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        atoms.get_dist()


def test_require_orthorhombic_returns_diagonal():
    cell = np.diag([10.0, 12.0, 14.0])
    np.testing.assert_allclose(require_orthorhombic(cell), [10.0, 12.0, 14.0])


def test_require_orthorhombic_names_the_caller():
    cell = np.array([[10.0, 0.0, 0.0], [1.0, 12.0, 0.0], [0.0, 0.0, 14.0]])
    with pytest.raises(NotImplementedError, match="my_function"):
        require_orthorhombic(cell, "my_function")


def test_get_coordination_number_matches_known_value(silicon_small):
    atoms = GlassAtoms(silicon_small)
    numbers = atoms.get_coordination_number("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    assert set(numbers) == {4}


def test_get_density_is_correct_for_silicon(silicon_small):
    """Crystalline silicon has a density of 2.329 g/cm^3."""
    assert GlassAtoms(silicon_small).get_density() == pytest.approx(2.329, abs=0.01)


def test_get_neighbors_returns_dict_keyed_by_species(silicon_small):
    """get_neighbors returns a dict of per-species neighbour lists, not a list of counts."""
    atoms = GlassAtoms(silicon_small)
    neighbors = atoms.get_neighbors("Si", SI_FIRST_SHELL_CUTOFF)
    assert isinstance(neighbors, dict)
    assert set(neighbors) == {"Si"}
    assert len(neighbors["Si"]) == len(silicon_small)
    assert all(len(entry) == 4 for entry in neighbors["Si"])


def test_get_neighbors_rejects_unknown_center(silicon_small):
    with pytest.raises(ValueError, match="not in the list of species"):
        GlassAtoms(silicon_small).get_neighbors("Ge", 3.0)


def test_get_neighbors_rejects_bad_cutoff_type(silicon_small):
    with pytest.raises(TypeError):
        GlassAtoms(silicon_small).get_neighbors("Si", "Auto")


def test_cross_pair_pdf_keeps_first_bin():
    """Only a like pair has zero-distance self-pairs to discard.

    A cross-pair block of the distance matrix has no diagonal, so blanking its first
    bin would drop a genuine short contact.
    """
    from ase import Atoms

    # One Si-O contact at 0.8 A, with 1.0 A wide bins, so it lands in bin 0.
    atoms = GlassAtoms(
        Atoms("SiO", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    )
    _, cross = atoms.get_pdf(["Si", "O"], rrange=10, nbin=10)
    assert cross[0] > 0


def test_like_pair_pdf_with_single_atom_is_zero():
    """A lone atom of a species has no pairs: zeros, not a division by zero."""
    from ase import Atoms

    atoms = GlassAtoms(
        Atoms("SiO", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True)
    )
    _, like = atoms.get_pdf(["Si", "Si"], rrange=10, nbin=10)
    assert np.all(like == 0.0)
