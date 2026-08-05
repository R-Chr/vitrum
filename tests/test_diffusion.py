"""Tests for vitrum.diffusion.

Coverage here is deliberately limited to the input guards: the MSD estimator itself uses
a single time origin, which is a known limitation rather than a bug (see
docs/vitrum/known_issues.md).
"""

import numpy as np
import pytest
from ase import Atoms

from vitrum.diffusion import Diffusion
from vitrum.trajectory_tools import unwrap_trajectory


def test_unwrap_follows_an_atom_across_a_boundary():
    """An atom drifting steadily must unwrap to a straight line, not fold back."""
    box = 4.0
    frames = [
        Atoms("Ar", positions=[[x, 0.0, 0.0]], cell=[box] * 3, pbc=True)
        for x in (3.5, 0.1, 0.7, 1.3)
    ]
    unwrapped = [a.get_positions()[0, 0] for a in unwrap_trajectory(frames)]
    np.testing.assert_allclose(unwrapped, [3.5, 4.1, 4.7, 5.3], atol=1e-9)


def test_unwrap_ignores_a_pure_cell_rescaling():
    """Under NPT the cell breathes; an atom fixed in the cell has not diffused.

    Its Cartesian position changes with the cell, so a step measured in Cartesian
    coordinates would report motion that never happened.
    """
    frames = [
        Atoms("Ar", positions=[[0.1 * box, 0.0, 0.0]], cell=[box] * 3, pbc=True)
        for box in (10.0, 9.0, 8.0)
    ]
    unwrapped = [a.get_positions()[0] for a in unwrap_trajectory(frames)]
    np.testing.assert_allclose(unwrapped, [unwrapped[0]] * 3, atol=1e-9)


def test_unwrap_measures_a_boundary_crossing_in_the_current_cell():
    """A crossing under a changing cell must be measured against the cell it happened in.

    The atom sits at fractional 0.95, then at 0.0333 after the cell has shrunk to 6 A:
    it stepped forward by 0.0833 of a cell, i.e. 0.5 A, through the boundary.
    """
    frames = [
        Atoms("Ar", positions=[[9.5, 0.0, 0.0]], cell=[10.0] * 3, pbc=True),
        Atoms("Ar", positions=[[0.2, 0.0, 0.0]], cell=[6.0] * 3, pbc=True),
    ]
    displacement = np.diff([a.get_positions()[0, 0] for a in unwrap_trajectory(frames)])
    assert displacement[0] == pytest.approx(0.5)


@pytest.fixture
def short_trajectory():
    """A 10-frame trajectory of one atom drifting along x, already unwrapped."""
    frames = [
        Atoms("Ar", positions=[[float(i) * 0.1, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
        for i in range(10)
    ]
    return frames, list(np.arange(10, dtype=float))


def test_diffusion_coef_rejects_oversized_skip(short_trajectory):
    """skip_first must not hand an empty slice to linregress.

    The default of 100 exceeds the length of any short trajectory, and scipy's own
    error for that ("Inputs must not be empty.") says nothing about which knob caused it.
    """
    trajectory, times = short_trajectory
    diffusion = Diffusion(trajectory, times, wrapped=False)

    with pytest.raises(ValueError, match="skip_first"):
        diffusion.get_diffusion_coef()

    with pytest.raises(ValueError, match="skip_first"):
        diffusion.get_diffusion_coef(skip_first=8)


def test_diffusion_coef_works_within_bounds(short_trajectory):
    trajectory, times = short_trajectory
    diffusion = Diffusion(trajectory, times, wrapped=False)
    coefficients = diffusion.get_diffusion_coef(skip_first=2)
    assert coefficients.shape == (2,)  # total, plus one species
    assert np.all(np.isfinite(coefficients))


def test_mismatched_sample_times_are_rejected(short_trajectory):
    trajectory, times = short_trajectory
    with pytest.raises(ValueError, match="one-to-one"):
        Diffusion(trajectory, times[:-2], wrapped=False)
