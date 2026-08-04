"""Tests for vitrum.coordination against crystalline silicon's known coordination."""

import numpy as np
import pytest

from vitrum.coordination import Coordination

# Between the 1st (2.3517 A) and 2nd (3.840 A) neighbour shells in diamond Si.
SI_FIRST_SHELL_CUTOFF = 3.0


def test_silicon_coordination_is_exactly_four(silicon_diamond):
    """Every atom in the diamond structure has exactly 4 nearest neighbours."""
    coordination = Coordination([silicon_diamond])
    distribution = coordination.get_coordination_numbers(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF
    )
    assert distribution == {4: pytest.approx(1.0)}


def test_silicon_second_shell_coordination_is_sixteen(silicon_diamond):
    """Cutoff past the second shell picks up 4 + 12 = 16 neighbours."""
    coordination = Coordination([silicon_diamond])
    distribution = coordination.get_coordination_numbers("Si", "Si", cutoff=4.2)
    assert distribution == {16: pytest.approx(1.0)}


def test_auto_cutoff_finds_first_shell(silicon_diamond):
    """The "Auto" cutoff should land between the first and second shells."""
    coordination = Coordination([silicon_diamond])
    distribution = coordination.get_coordination_numbers("Si", "Si", cutoff="Auto")
    assert distribution == {4: pytest.approx(1.0)}


def test_unexpected_cutoff_string_raises_value_error(silicon_diamond):
    """A bad cutoff must produce a useful error, not UnboundLocalError.

    The string branch matches "Auto" exactly, so anything else has to be rejected
    explicitly rather than falling through with `cutoffs` unbound.
    """
    coordination = Coordination([silicon_diamond])
    with pytest.raises(ValueError, match="Invalid cutoff"):
        coordination.get_coordination_numbers("Si", "Si", cutoff="auto")


def test_mismatched_cutoff_list_length_raises(silicon_diamond):
    coordination = Coordination([silicon_diamond])
    with pytest.raises(ValueError, match="must match"):
        coordination.get_coordination_numbers("Si", "Si", cutoff=[2.0, 3.0])


def test_missing_species_raises(silicon_diamond):
    coordination = Coordination([silicon_diamond])
    with pytest.raises(ValueError, match="not found in structure"):
        coordination.get_coordination_numbers("Ge", "Si", cutoff=3.0)


def test_coordination_fractions_sum_to_one(random_gas):
    coordination = Coordination([random_gas])
    distribution = coordination.get_coordination_numbers("Si", "O", cutoff=4.0)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_silicon_tetrahedral_angle(silicon_diamond):
    """The Si-Si-Si angle distribution must peak at the tetrahedral 109.47 degrees.

    An explicit range is required: a perfect crystal gives a delta function, and
    np.histogram cannot bin data whose min and max coincide.
    """
    coordination = Coordination([silicon_diamond])
    angles, distribution = coordination.get_angle_distribution(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, range=(0.0, 180.0)
    )
    assert angles[np.argmax(distribution)] == pytest.approx(109.47, abs=2.0)
