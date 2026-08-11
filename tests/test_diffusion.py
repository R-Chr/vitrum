"""Tests for vitrum.diffusion.

Coverage here is deliberately limited to the input guards: the MSD estimator itself uses
a single time origin, which is a known limitation rather than a bug (see
docs/vitrum/known_issues.md).
"""

import numpy as np
import pytest
from ase import Atoms

from vitrum.diffusion import Diffusion
from vitrum.trajectory_tools import get_high_low_displacement_index, unwrap_trajectory


def test_unwrap_follows_an_atom_across_a_boundary():
    """An atom drifting steadily must unwrap to a straight line, not fold back."""
    box = 4.0
    frames = [Atoms("Ar", positions=[[x, 0.0, 0.0]], cell=[box] * 3, pbc=True) for x in (3.5, 0.1, 0.7, 1.3)]
    unwrapped = [a.get_positions()[0, 0] for a in unwrap_trajectory(frames)]
    np.testing.assert_allclose(unwrapped, [3.5, 4.1, 4.7, 5.3], atol=1e-9)


def test_unwrap_ignores_a_pure_cell_rescaling():
    """Under NPT the cell breathes; an atom fixed in the cell has not diffused.

    Its Cartesian position changes with the cell, so a step measured in Cartesian
    coordinates would report motion that never happened.
    """
    frames = [Atoms("Ar", positions=[[0.1 * box, 0.0, 0.0]], cell=[box] * 3, pbc=True) for box in (10.0, 9.0, 8.0)]
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


def test_high_and_low_displacement_atoms_are_split_by_how_far_they_moved():
    """Four Na, moved by 0, 1, 2 and 3 A. A quarter is one atom: the one that did not move.

    The two index arrays must partition that species and nothing else -- the O atoms are
    interleaved to catch a routine that indexed into its own selection instead of the frame.
    """
    symbols = "NaONaONaONaO"  # eight atoms: Na at 0, 2, 4, 6
    initial = Atoms(symbols, positions=[[0.0, 0.0, 0.0]] * 8, cell=[50] * 3, pbc=True)
    moved = initial.copy()
    positions = moved.get_positions()
    for atom, shift in zip(range(0, 8, 2), (0.0, 1.0, 2.0, 3.0)):
        positions[atom, 0] = shift
    moved.set_positions(positions)

    low, high = get_high_low_displacement_index(initial, moved, "Na", percentage=0.25)
    assert low.tolist() == [0]
    assert sorted(high.tolist()) == [2, 4, 6]


def test_displacement_split_ignores_other_species():
    """Only the target species is ranked, so a wildly moving O must not appear."""
    initial = Atoms("NaNaO", positions=[[0.0, 0.0, 0.0]] * 3, cell=[50] * 3, pbc=True)
    moved = initial.copy()
    moved.set_positions([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

    low, high = get_high_low_displacement_index(initial, moved, "Na", percentage=0.5)
    assert set(low.tolist()) | set(high.tolist()) == {0, 1}


@pytest.fixture
def short_trajectory():
    """A 10-frame trajectory of one atom drifting along x, already unwrapped."""
    frames = [Atoms("Ar", positions=[[float(i) * 0.1, 0.0, 0.0]], cell=[10, 10, 10], pbc=True) for i in range(10)]
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


def test_an_already_unwrapped_trajectory_is_rejected(short_trajectory):
    """Unwrapping an unwrapped trajectory folds real displacements back into the cell.

    `short_trajectory` drifts to 0.9 A in a 10 A cell, so shifting it three cells out is
    what an unwrapped trajectory looks like: atoms whole cells away, not the fraction of
    one that a wrapped dump can reach at a boundary.
    """
    trajectory, times = short_trajectory
    far_out = [atoms.copy() for atoms in trajectory]
    for atoms in far_out:
        atoms.positions += 30.0

    with pytest.raises(ValueError, match="outside it"):
        Diffusion(far_out, times)


def test_a_wrapped_trajectory_touching_the_boundary_is_accepted(short_trajectory):
    """An MD code writes wrapped coordinates that can land just past a face.

    A hair outside is not the same as whole cells outside, and only the latter means the
    trajectory was already unwrapped.
    """
    trajectory, times = short_trajectory
    at_the_edge = [atoms.copy() for atoms in trajectory]
    for atoms in at_the_edge:
        atoms.positions -= 0.05  # 0.005 of the 10 A cell, as a rounded-down dump would

    assert len(Diffusion(at_the_edge, times).trajectory) == len(trajectory)
