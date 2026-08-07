"""Tests that public functions return what their annotations and docstrings claim.

They are cheap, and they pin the one thing a caller reads before anything else: the
shape of what comes back.
"""

import numpy as np
import pytest

from vitrum.comparison import r_chi
from vitrum.io_helpers import correct_atom_types, get_density
from vitrum.scattering import Scattering


def test_weighted_partial_structure_factors_returns_dict(random_gas):
    """Documented and annotated as a dict of label -> W_ij * S_ij(Q)."""
    scattering = Scattering(random_gas, disable_progress=True)
    result = scattering.get_weighted_partial_structure_factors()
    assert isinstance(result, dict)
    assert set(result) == {"O-O", "O-Si", "Si-Si"}
    for value in result.values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (scattering.nbin,)


def test_weighted_partials_sum_to_total_structure_factor(random_gas):
    """The docstring claims the weighted partials sum to the total S(Q)."""
    scattering = Scattering(random_gas, disable_progress=True)
    partials = scattering.get_weighted_partial_structure_factors(type="neutron")
    total = sum(partials.values())
    np.testing.assert_allclose(
        total, scattering.get_structure_factor(type="neutron"), rtol=1e-6
    )


def test_weighted_partials_rejects_bad_type(random_gas):
    scattering = Scattering(random_gas, disable_progress=True)
    with pytest.raises(ValueError):
        scattering.get_weighted_partial_structure_factors(type="approx_xray")


def test_correct_atom_types_mutates_in_place_and_returns_none(silicon_small):
    """Documented as returning None and modifying the Atoms objects in place."""
    atoms = silicon_small.copy()
    result = correct_atom_types([atoms], {14: "Ge"})
    assert result is None
    assert set(atoms.get_chemical_symbols()) == {"Ge"}


def test_get_density_returns_a_float_in_g_per_cm3(silicon_small):
    """Crystalline silicon has a density of 2.329 g/cm^3."""
    density = get_density(silicon_small)
    assert isinstance(density, float)
    assert density == pytest.approx(2.329, abs=0.01)


def test_r_chi_returns_four_tuple():
    """Documented as (rchi, common_x, y1, y2)."""
    x = np.linspace(0, 10, 50)
    f1 = {"x": x, "y": np.sin(x)}
    f2 = {"x": x, "y": np.sin(x)}
    result = r_chi(f1, f2)
    assert isinstance(result, tuple)
    assert len(result) == 4
    rchi, common_x, y1, y2 = result
    assert isinstance(rchi, float)
    # Identical functions are a perfect match
    assert rchi == pytest.approx(0.0, abs=1e-12)
    for array in (common_x, y1, y2):
        assert isinstance(array, np.ndarray)


def test_r_chi_raises_without_overlap():
    f1 = {"x": np.linspace(0, 1, 10), "y": np.ones(10)}
    f2 = {"x": np.linspace(5, 6, 10), "y": np.ones(10)}
    with pytest.raises(ValueError, match="No overlapping x-range"):
        r_chi(f1, f2)
