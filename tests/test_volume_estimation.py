"""Tests for vitrum.volume_estimation: the oxidation-state guess the radii are built on.

Guessing exactly is combinatorial, so large compositions fall back to a rounded stand-in that
is not always charge-balanceable. What is checked here is that the fallback is reserved for
compositions that are genuinely expensive, not merely large.
"""

import time
import warnings

import pytest
from pymatgen.core import Composition

from vitrum.volume_estimation import get_packing_radii, guess_oxi_states


def test_large_cell_of_single_state_elements_is_guessed_exactly():
    """287 atoms, but Ca, Al and O have one known oxidation state each, so it is cheap."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert guess_oxi_states(Composition("Ca49Al36Si33O169")) == {"Ca": 2, "Al": 3, "Si": 4, "O": -2}


def test_expensive_composition_falls_back_without_hanging():
    """P has six known states: guessing 37 of them exactly takes minutes, so it must not try."""
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        guess_oxi_states(Composition("Ca49P37O151"))
    assert time.time() - start < 10


def test_unbalanced_composition_warns_and_falls_back_to_covalent_radii():
    composition = Composition("CuZr")
    with pytest.warns(UserWarning, match="oxidation states"):
        assert guess_oxi_states(composition) is None
    assert get_packing_radii(["Cu", "Zr"], composition, source="ionic") == pytest.approx([1.32, 1.75])


def test_supplied_oxidation_states_are_used_as_given():
    """Fe2O3 guesses Fe3+; passing Fe2+ must give that ion's (larger) radius instead."""
    composition = Composition("Fe2O3")
    guessed = get_packing_radii(["Fe"], composition, source="ionic")
    supplied = get_packing_radii(["Fe"], composition, source="ionic", oxi={"Fe": 2, "O": -2})
    assert supplied > guessed
