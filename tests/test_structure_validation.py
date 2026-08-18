"""Tests for vitrum.structure_validation."""

import pytest

from vitrum.structure_validation import dimer_checker, homogeneity_checker


def test_single_dimer_not_flagged_at_threshold_one(one_dimer):
    """One real dimer must not trip a threshold of one.

    The check works from a symmetric distance matrix, which contains each dimer twice;
    `num_allowed` counts dimers, not matrix entries.
    """
    assert dimer_checker(one_dimer, num_allowed=1) is False


def test_single_dimer_flagged_at_threshold_zero(one_dimer):
    assert dimer_checker(one_dimer, num_allowed=0) is True


def test_dimer_across_periodic_boundary_is_detected(boundary_dimer):
    """A dimer spanning the cell boundary must be found.

    Its two atoms sit at opposite faces, so they are a full cell length apart in raw
    Cartesian distance and only close under the minimum image convention.
    """
    assert dimer_checker(boundary_dimer, num_allowed=0) is True


def test_no_dimers_in_well_separated_structure(uniform_lattice):
    assert dimer_checker(uniform_lattice, bond_length=2.0, num_allowed=0) is False


@pytest.mark.parametrize("grid_density", [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (2, 3, 4)])
def test_homogeneity_checker_handles_any_grid_density(uniform_random, grid_density):
    """Every grid density must work, not just 3x3x3.

    The checker slides the grid and re-bins the atoms that fall below the lower edge,
    which has to land them in the last box along that axis -- an index that depends on
    the grid density.
    """
    assert isinstance(homogeneity_checker(uniform_random, grid_density), bool)


@pytest.mark.parametrize("grid_density", [(2, 2, 2), (3, 3, 3), (4, 4, 4)])
def test_uniform_random_structure_is_homogeneous(uniform_random, grid_density):
    assert homogeneity_checker(uniform_random, grid_density) is True


@pytest.mark.parametrize("grid_density", [(2, 2, 2), (3, 3, 3), (4, 4, 4)])
def test_phase_separated_structure_is_flagged(phase_separated, grid_density):
    assert homogeneity_checker(phase_separated, grid_density) is False


def test_homogeneity_checker_rejects_triclinic(triclinic_atoms):
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        homogeneity_checker(triclinic_atoms, (3, 3, 3))
