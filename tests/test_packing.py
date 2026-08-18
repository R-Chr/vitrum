"""Tests for vitrum.packing: the charge-aware side of the overlap resolution.

The purely geometric path is a smoke test in test_packaging.py; what is checked here is that
`charge_ordering` separates like-charged ions without pulling any other pair into an overlap,
and that leaving it off changes nothing.
"""

import warnings

import numpy as np
import pytest

from vitrum.packing import get_random_packed


def min_distances(atoms):
    """{(symbol, symbol): shortest distance} over every species pair, self-pairs included."""
    symbols = np.array(atoms.get_chemical_symbols())
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    species = sorted(set(symbols))
    return {
        (a, b): distances[np.ix_(symbols == a, symbols == b)].min() for i, a in enumerate(species) for b in species[i:]
    }


def test_charge_ordering_is_off_by_default():
    default = get_random_packed("SiO2", target_atoms=96, seed=0)
    explicit = get_random_packed("SiO2", target_atoms=96, seed=0, charge_ordering=0.0)
    assert np.allclose(default.get_positions(), explicit.get_positions())


def test_charge_ordering_separates_cations_without_new_overlaps():
    plain = min_distances(get_random_packed("SiO2", target_atoms=96, seed=0))
    ordered = min_distances(get_random_packed("SiO2", target_atoms=96, seed=0, charge_ordering=1.0))

    assert plain[("Si", "Si")] < 2.0  # the defect this guards against
    assert ordered[("Si", "Si")] > 2.5
    assert ordered[("O", "Si")] > 1.6  # the ionic contact distance, not pushed through
    assert min(ordered.values()) > 1.5


def test_charge_ordering_converges_on_a_cation_rich_glass():
    """Na2Si2O5 is 4 cations to 5 anions: enough of them that an uncapped shell cannot fit."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Cell packing not converged")
        atoms = get_random_packed("Na2Si2O5", target_atoms=108, seed=0, charge_ordering=1.0)
    assert min(min_distances(atoms).values()) > 1.5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_charge_ordering_is_a_no_op_without_oxidation_states():
    atoms = get_random_packed("CuZr", target_atoms=64, seed=0, charge_ordering=1.0)
    assert len(atoms) == 64
