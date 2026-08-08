"""Tests for vitrum.voids.

Ground truth is the volume of a sphere: a single atom in a large cell excludes
4/3 pi r^3 and nothing else, so the occupancy grid has an analytic answer to converge to.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from vitrum.voids import VoidAnalysis

# Small enough that a 0.15 A grid stays cheap, large enough that the 1.06 A argon sphere
# never reaches a face even at twice its radius.
BOX = 12.0
AR_RADIUS = covalent_radii[atomic_numbers["Ar"]]


@pytest.fixture
def lone_atom():
    """One argon at the origin of a 12 A cube: one atom, one cavity, one known volume."""
    return Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[BOX] * 3, pbc=True)


def _occupied_fraction(atoms, **kwargs):
    analysis = VoidAnalysis(atoms, **kwargs)
    analysis.calculate(grid_spacing=0.15)
    return 1 - analysis.get_free_volume_fraction()


def test_occupied_volume_of_one_atom_is_its_sphere(lone_atom):
    """The excluded volume is 4/3 pi r^3 out of the cell, to within the grid resolution."""
    expected = (4 / 3) * np.pi * AR_RADIUS**3 / BOX**3
    assert _occupied_fraction(lone_atom) == pytest.approx(expected, rel=0.05)


def test_a_finer_grid_converges_on_the_sphere(lone_atom):
    """Voxelising a sphere under-counts it, and less so as the grid tightens."""
    expected = (4 / 3) * np.pi * AR_RADIUS**3 / BOX**3
    errors = []
    for spacing in (0.5, 0.25, 0.15):
        analysis = VoidAnalysis(lone_atom)
        analysis.calculate(grid_spacing=spacing)
        errors.append(abs((1 - analysis.get_free_volume_fraction()) - expected))
    assert errors[0] > errors[1] > errors[2]


def test_probe_radius_inflates_the_exclusion(lone_atom):
    """A probe of radius p sees a sphere of r + p, which is the point of the parameter."""
    expected = (4 / 3) * np.pi * (AR_RADIUS + 1.0) ** 3 / BOX**3
    assert _occupied_fraction(lone_atom, probe_radius=1.0) == pytest.approx(expected, rel=0.05)


def test_radii_scaling_scales_the_radius_not_the_volume(lone_atom):
    """Doubling the radius must give eight times the volume, not twice."""
    single = _occupied_fraction(lone_atom)
    doubled = _occupied_fraction(lone_atom, radii_scaling=2.0)
    assert doubled == pytest.approx(8 * single, rel=0.05)


def test_radii_overrides_replace_the_tabulated_radius(lone_atom):
    """An override is a base radius in Angstrom, taken instead of the covalent one."""
    expected = (4 / 3) * np.pi * 2.0**3 / BOX**3
    assert _occupied_fraction(lone_atom, radii_overrides={"Ar": 2.0}) == pytest.approx(
        expected, rel=0.05
    )


def test_the_free_space_around_one_atom_is_a_single_cavity(lone_atom):
    """It is all connected through the periodic boundary, so it must not fragment."""
    analysis = VoidAnalysis(lone_atom)
    cavities = analysis.calculate(grid_spacing=0.5)
    assert len(cavities) == 1
    assert cavities[0].volume() == pytest.approx(BOX**3, rel=0.01)


def test_a_dense_structure_has_less_free_volume_than_a_sparse_one(silicon_small, lone_atom):
    """The headline number has to respond to how full the cell actually is."""
    assert _occupied_fraction(silicon_small) > _occupied_fraction(lone_atom)


def test_the_melt_has_free_volume_but_is_mostly_full(sodium_silicate_frame):
    """A real glass is neither empty nor close-packed; both extremes would be a bug."""
    analysis = VoidAnalysis(sodium_silicate_frame)
    analysis.calculate(grid_spacing=0.4)
    assert 0.0 < analysis.get_free_volume_fraction() < 1.0


def test_results_are_unavailable_before_calculate(lone_atom):
    analysis = VoidAnalysis(lone_atom)
    with pytest.raises(ValueError, match="not been calculated"):
        analysis.get_free_volume_fraction()
    with pytest.raises(ValueError, match="not been calculated"):
        analysis.get_cavity_size_distribution()


def test_cavity_size_distribution_rejects_an_unknown_quantity(lone_atom):
    analysis = VoidAnalysis(lone_atom)
    analysis.calculate(grid_spacing=0.5)
    with pytest.raises(ValueError, match="Unknown value for `by`"):
        analysis.get_cavity_size_distribution(by="diameter")


def test_triclinic_cell_is_rejected(triclinic_atoms):
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        VoidAnalysis(triclinic_atoms)


def test_the_input_structure_is_not_mutated(sodium_silicate_frame):
    """VoidAnalysis wraps the atoms it is given; it must wrap its own copy."""
    before = sodium_silicate_frame.get_positions().copy()
    VoidAnalysis(sodium_silicate_frame).calculate(grid_spacing=0.5)
    np.testing.assert_array_equal(sodium_silicate_frame.get_positions(), before)
