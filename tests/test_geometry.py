"""Tests for vitrum.geometry: the orthorhombic guard, distances and partial PDFs."""

import numpy as np
import pytest
from ase import Atoms

from vitrum.geometry import (
    _minkowski_basis,
    distance_matrix,
    minimum_image_limit,
    partial_pdf,
    peak_metrics,
    perpendicular_widths,
    require_orthorhombic,
)


def test_distance_matrix_matches_ase_minimum_image(silicon_small):
    """The numba distance kernel must agree with ASE's own MIC distances."""
    np.testing.assert_allclose(distance_matrix(silicon_small), silicon_small.get_all_distances(mic=True), atol=1e-8)


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
    np.testing.assert_allclose(distance_matrix(unwrapped), distance_matrix(wrapped), atol=1e-8)


# Rounding the fractional separation alone does not find the nearest image once the axes
# are far from perpendicular, so the shapes here are chosen to be progressively worse for
# it: a hexagonal cell, a monoclinic one, one sheared beyond any physical cell, and one
# long thin cell whose shortest width is a fraction of its longest axis.
GENERAL_CELLS = {
    "hexagonal": [[12.0, 0.0, 0.0], [-6.0, 10.392, 0.0], [0.0, 0.0, 19.0]],
    "monoclinic": [[10.0, 0.0, 0.0], [0.0, 13.0, 0.0], [-4.2, 0.0, 11.0]],
    "extreme shear": [[12.0, 0.0, 0.0], [23.0, 12.0, 0.0], [17.0, 19.0, 12.0]],
    "long thin": [[40.0, 0.0, 0.0], [3.0, 6.0, 0.0], [2.0, 1.0, 6.5]],
}


@pytest.mark.parametrize("cell", GENERAL_CELLS.values(), ids=list(GENERAL_CELLS))
def test_distance_matrix_matches_ase_for_general_cells(cell):
    """Every cell shape must agree with ASE, for positions well outside the box.

    The positions span three cell lengths in each direction, so the minimum image has to
    hold for separations of several cells and not merely within one, exactly as it must
    for the unwrapped coordinates a LAMMPS dump stores.
    """
    cell = np.asarray(cell, dtype=float)
    rng = np.random.default_rng(0)
    positions = (rng.random((60, 3)) * 3 - 1) @ cell
    atoms = Atoms("Si60", positions=positions, cell=cell, pbc=True)
    np.testing.assert_allclose(distance_matrix(atoms), atoms.get_all_distances(mic=True), atol=1e-8)


def test_orthorhombic_cells_skip_the_image_search():
    """Perpendicular axes need no images searched; the shortcut is most of the kernel's cost.

    Correctness of the shortcut is covered by every orthorhombic distance test, which all
    go through it. This pins that it is actually taken.
    """
    assert _minkowski_basis(np.diag([9.0, 14.0, 11.0]), [True] * 3)[2].shape == (1, 3)
    assert _minkowski_basis(GENERAL_CELLS["hexagonal"], [True] * 3)[2].shape == (27, 3)


def test_perpendicular_widths_are_the_diagonal_for_an_orthorhombic_cell():
    """The general bound must reduce to the old one, or every default rrange shifts."""
    np.testing.assert_allclose(perpendicular_widths(np.diag([10.0, 12.0, 14.0])), [10.0, 12.0, 14.0])


def test_perpendicular_widths_shrink_under_shear():
    """Shearing an axis leaves its length alone but narrows the cell across it."""
    cell = np.array([[10.0, 0.0, 0.0], [6.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    widths = perpendicular_widths(cell)
    # The b axis is still sqrt(6^2 + 10^2) = 11.66 long, but the a-c faces it spans are
    # only 10 apart; it is the a direction that narrows, to V / |b x c| = 1000 / 116.6.
    np.testing.assert_allclose(widths, [1000 / np.linalg.norm(np.cross(cell[1], cell[2])), 10.0, 10.0])
    assert widths.min() < 10.0


def test_minimum_image_limit_is_half_the_width_for_an_orthorhombic_cell():
    """Reduction is a no-op on an orthorhombic cell, so the limit is just half the shortest."""
    assert minimum_image_limit(np.diag([10.0, 12.0, 14.0]), [True] * 3) == pytest.approx(5.0)


def test_minimum_image_limit_never_exceeds_either_basis():
    """The limit must hold for the cell as given *and* for its Minkowski reduction.

    Reduction shortens the basis vectors, but shortening one can narrow the cell across a
    face -- measured at up to 10% narrower over random skewed cells. Taking only the
    unreduced widths would let `Scattering` accept an rrange that `cell_list_pair_counts`,
    which grids on the reduced basis, refuses. With no fallback backend that is a crash, so
    the two have to agree by construction rather than by luck.
    """
    rng = np.random.default_rng(0)
    narrowed = 0
    for _ in range(2000):
        cell = np.eye(3) * 10 + rng.normal(0, 6, (3, 3))
        if abs(np.linalg.det(cell)) < 50:
            continue
        limit = minimum_image_limit(cell, [True] * 3)
        unreduced = perpendicular_widths(cell).min() / 2
        reduced = perpendicular_widths(_minkowski_basis(cell, [True] * 3)[0]).min() / 2
        assert limit <= unreduced + 1e-12 and limit <= reduced + 1e-12
        narrowed += reduced < unreduced - 1e-9
    assert narrowed > 0, "no cell in this sample had a narrower reduction; the test proves nothing"


def test_require_orthorhombic_still_rejects_triclinic(triclinic_atoms):
    """`distance_matrix` accepts a general cell now, but the void/unwrap guard must not."""
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        require_orthorhombic(triclinic_atoms.get_cell(), "build_void_grid")


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
        distance_matrix(atoms),
        atoms.get_chemical_symbols(),
        atoms.get_volume(),
        ["Si", "O"],
        rrange=10,
        nbin=10,
    )
    assert cross[0] > 0


def test_like_pair_pdf_with_single_atom_is_zero():
    """A lone atom of a species has no pairs: zeros, not a division by zero."""
    atoms = _si_o_pair()
    _, like = partial_pdf(
        distance_matrix(atoms),
        atoms.get_chemical_symbols(),
        atoms.get_volume(),
        ["Si", "Si"],
        rrange=10,
        nbin=10,
    )
    assert np.all(like == 0.0)


def test_partial_pdf_returns_zeros_for_absent_species(silicon_small):
    """A species that is not in the structure has no pairs to bin."""
    _, absent = partial_pdf(
        distance_matrix(silicon_small),
        silicon_small.get_chemical_symbols(),
        silicon_small.get_volume(),
        ["Si", "Ge"],
        nbin=50,
    )
    assert np.all(absent == 0.0)


def test_partial_pdf_accepts_numpy_integer_atomic_numbers(silicon_small):
    """Atomic numbers taken from ASE arrive as numpy integers, not Python ints."""
    distances = distance_matrix(silicon_small)
    volume = silicon_small.get_volume()
    numbers = np.unique(silicon_small.get_atomic_numbers())
    _, from_numpy = partial_pdf(distances, silicon_small.get_atomic_numbers(), volume, [numbers[0], numbers[0]])
    _, from_symbol = partial_pdf(distances, silicon_small.get_chemical_symbols(), volume, ["Si", "Si"])
    np.testing.assert_allclose(from_numpy, from_symbol)


def test_partial_pdf_indices_override_the_pair(silicon_small):
    """Explicit index arrays select the same atoms as matching on symbols."""
    distances = distance_matrix(silicon_small)
    volume = silicon_small.get_volume()
    all_si = np.arange(len(silicon_small))
    _, by_index = partial_pdf(distances, None, volume, ["ignored", "ignored"], indices=[all_si, all_si])
    _, by_symbol = partial_pdf(distances, silicon_small.get_chemical_symbols(), volume, ["Si", "Si"])
    np.testing.assert_allclose(by_index, by_symbol)


def test_peak_metrics_recovers_a_known_gaussian():
    """A Gaussian of width sigma has FWHM = 2*sqrt(2*ln2)*sigma, and its height and centre
    are what it was built with."""
    x = np.linspace(0.0, 10.0, 2001)
    sigma, centre, amplitude = 0.4, 3.0, 2.5
    y = amplitude * np.exp(-((x - centre) ** 2) / (2 * sigma**2))

    position, fwhm, height = peak_metrics(x, y)
    assert position == pytest.approx(centre, abs=x[1] - x[0])
    assert fwhm == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, rel=1e-3)
    assert height == pytest.approx(amplitude, rel=1e-6)


def test_peak_metrics_takes_the_first_peak_and_the_window_selects_another():
    """Two well-separated peaks: the default finds the first, a window finds the second."""
    x = np.linspace(0.0, 10.0, 2001)
    y = np.exp(-((x - 2.0) ** 2) / 0.08) + 3 * np.exp(-((x - 7.0) ** 2) / 0.08)

    assert peak_metrics(x, y)[0] == pytest.approx(2.0, abs=1e-2)
    assert peak_metrics(x, y, window=(5.0, 10.0))[0] == pytest.approx(7.0, abs=1e-2)


def test_peak_metrics_on_a_monotonic_function_raises():
    x = np.linspace(0.0, 10.0, 200)
    with pytest.raises(ValueError, match="No local maximum"):
        peak_metrics(x, x**2)


def test_peak_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same shape"):
        peak_metrics(np.arange(10.0), np.arange(9.0))
