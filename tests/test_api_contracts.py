"""Tests that public functions return what their annotations and docstrings claim.

They are cheap, and they pin the one thing a caller reads before anything else: the
shape of what comes back.
"""

import numpy as np
import pytest

from vitrum.comparison import r_chi
from vitrum.io_helpers import correct_atom_types, get_LAMMPS_dump_timesteps, get_density
from vitrum.scattering import Scattering


@pytest.mark.parametrize(
    "dump, expected",
    [
        ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: TIMESTEP\n1000\n", [0, 1000]),
        ("ITEM: TIMESTEP\n7", [7]),  # no trailing newline
        ("ITEM: TIMESTEP\n0\nITEM: TIMESTEP\n", [0]),  # truncated on the header
        ("junk\nmore junk\n", []),
        ("", []),
    ],
)
def test_lammps_dump_timesteps(tmp_path, dump, expected):
    """The header line is skipped and the line after it is the timestep."""
    path = tmp_path / "md.lammpstrj"
    path.write_text(dump, encoding="utf-8")
    assert get_LAMMPS_dump_timesteps(str(path)) == expected


@pytest.fixture(scope="module")
def glass_scattering(sodium_silicate_frame):
    return Scattering(sodium_silicate_frame, disable_progress=True)


def test_weighted_partial_structure_factors_returns_dict(glass_scattering):
    """Documented and annotated as a dict of label -> W_ij * S_ij(Q)."""
    result = glass_scattering.get_weighted_partial_structure_factors()
    assert isinstance(result, dict)
    # Three species give the six unordered pairs.
    assert set(result) == {"Na-Na", "Na-O", "Na-Si", "O-O", "O-Si", "Si-Si"}
    for value in result.values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (glass_scattering.nbin,)


@pytest.mark.parametrize("type", ["neutron", "xray", "approx_xray"])
def test_weighted_partials_sum_to_total_for_every_weighting(glass_scattering, type):
    """xray weights are Q-dependent arrays, so a wrong summation axis in the normalisation
    would still return a plausibly shaped S(Q)."""
    partials = glass_scattering.get_weighted_partial_structure_factors(type=type)
    np.testing.assert_allclose(
        sum(partials.values()), glass_scattering.get_structure_factor(type=type), rtol=1e-6
    )


def test_weighted_partials_rejects_bad_type(glass_scattering):
    with pytest.raises(ValueError, match="Invalid type"):
        glass_scattering.get_weighted_partial_structure_factors(type="not-a-type")


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
