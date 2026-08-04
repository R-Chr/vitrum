"""Tests for vitrum.diffusion.

Coverage here is deliberately limited to the input guards: the MSD estimator itself uses
a single time origin, which is a known limitation rather than a bug (see
docs/vitrum/known_issues.md).
"""

import numpy as np
import pytest
from ase import Atoms

from vitrum.diffusion import Diffusion


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
