"""Tests for vitrum.geometry: the orthorhombic guard, distances and partial PDFs."""

import numpy as np
import pytest
from ase import Atoms

from vitrum.geometry import distance_matrix, partial_pdf, require_orthorhombic


def test_distance_matrix_matches_ase_minimum_image(silicon_small):
    """The numba distance kernel must agree with ASE's own MIC distances."""
    np.testing.assert_allclose(
        distance_matrix(silicon_small), silicon_small.get_all_distances(mic=True), atol=1e-8
    )


@pytest.mark.parametrize("x", [12.0, 21.5, 25.0, -3.0, -18.0])
def test_distance_matrix_matches_ase_for_positions_outside_the_cell(x):
    """Unwrapped coordinates must give the same distance as wrapped ones.

    Trajectory formats such as LAMMPS dumps routinely store positions several cell
    lengths outside the box, so the minimum image has to hold for separations beyond one
    cell length, not just within it.
    """
    box = 10.0
    atoms = Atoms("H2", positions=[[0.5, 0, 0], [x, 0, 0]], cell=[box] * 3, pbc=True)
    assert distance_matrix(atoms)[0, 1] == pytest.approx(atoms.get_distance(0, 1, mic=True))


def test_distance_matrix_is_unchanged_by_wrapping(silicon_small):
    """Wrapping the structure into the cell must not move any distance."""
    unwrapped = silicon_small.copy()
    unwrapped.positions += unwrapped.get_cell().lengths() * np.array([1.0, -2.0, 3.0])
    wrapped = unwrapped.copy()
    wrapped.wrap()
    np.testing.assert_allclose(
        distance_matrix(unwrapped), distance_matrix(wrapped), atol=1e-8
    )


def test_distance_matrix_is_symmetric(silicon_small):
    distances = distance_matrix(silicon_small)
    np.testing.assert_allclose(distances, distances.T, atol=1e-12)


def test_distance_matrix_rejects_triclinic(triclinic_atoms):
    """A triclinic cell must raise rather than silently give wrong distances."""
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        distance_matrix(triclinic_atoms)


def test_distance_matrix_names_the_caller(triclinic_atoms):
    with pytest.raises(NotImplementedError, match="my_analysis"):
        distance_matrix(triclinic_atoms, "my_analysis")


def test_require_orthorhombic_returns_diagonal():
    cell = np.diag([10.0, 12.0, 14.0])
    np.testing.assert_allclose(require_orthorhombic(cell), [10.0, 12.0, 14.0])


def test_require_orthorhombic_names_the_caller():
    cell = np.array([[10.0, 0.0, 0.0], [1.0, 12.0, 0.0], [0.0, 0.0, 14.0]])
    with pytest.raises(NotImplementedError, match="my_function"):
        require_orthorhombic(cell, "my_function")


def _si_o_pair():
    """One Si-O contact at 0.8 A, in a 10 A cube."""
    return Atoms("SiO", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True)


def test_cross_pair_pdf_keeps_first_bin():
    """Only a like pair has zero-distance self-pairs to discard.

    A cross-pair block of the distance matrix has no diagonal, so blanking its first
    bin would drop a genuine short contact.
    """
    atoms = _si_o_pair()
    # With 1.0 A wide bins, the 0.8 A contact lands in bin 0.
    _, cross = partial_pdf(
        distance_matrix(atoms), atoms.get_chemical_symbols(), atoms.get_volume(),
        ["Si", "O"], rrange=10, nbin=10,
    )
    assert cross[0] > 0


def test_like_pair_pdf_with_single_atom_is_zero():
    """A lone atom of a species has no pairs: zeros, not a division by zero."""
    atoms = _si_o_pair()
    _, like = partial_pdf(
        distance_matrix(atoms), atoms.get_chemical_symbols(), atoms.get_volume(),
        ["Si", "Si"], rrange=10, nbin=10,
    )
    assert np.all(like == 0.0)


def test_partial_pdf_returns_zeros_for_absent_species(silicon_small):
    """A species that is not in the structure has no pairs to bin."""
    _, absent = partial_pdf(
        distance_matrix(silicon_small), silicon_small.get_chemical_symbols(),
        silicon_small.get_volume(), ["Si", "Ge"], nbin=50,
    )
    assert np.all(absent == 0.0)


def test_partial_pdf_accepts_numpy_integer_atomic_numbers(silicon_small):
    """Atomic numbers taken from ASE arrive as numpy integers, not Python ints."""
    distances = distance_matrix(silicon_small)
    volume = silicon_small.get_volume()
    numbers = np.unique(silicon_small.get_atomic_numbers())
    _, from_numpy = partial_pdf(
        distances, silicon_small.get_atomic_numbers(), volume, [numbers[0], numbers[0]]
    )
    _, from_symbol = partial_pdf(
        distances, silicon_small.get_chemical_symbols(), volume, ["Si", "Si"]
    )
    np.testing.assert_allclose(from_numpy, from_symbol)


def test_partial_pdf_indices_override_the_pair(silicon_small):
    """Explicit index arrays select the same atoms as matching on symbols."""
    distances = distance_matrix(silicon_small)
    volume = silicon_small.get_volume()
    all_si = np.arange(len(silicon_small))
    _, by_index = partial_pdf(
        distances, None, volume, ["ignored", "ignored"], indices=[all_si, all_si]
    )
    _, by_symbol = partial_pdf(
        distances, silicon_small.get_chemical_symbols(), volume, ["Si", "Si"]
    )
    np.testing.assert_allclose(by_index, by_symbol)
