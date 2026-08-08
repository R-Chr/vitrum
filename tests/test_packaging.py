"""Tests for the seam between required and optional dependencies.

`vitrum` imports structure_gen -> packing -> volume_estimation at package import time,
but volume_estimation's database backends need `atomate2` and `mp_api`, which are optional
extras. These tests run in a subprocess with those modules blocked, because a developer
machine that happens to have them installed cannot otherwise tell the two apart.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

# Installs a meta-path finder that makes the optional extras unimportable, so the
# subprocess sees what a clean `pip install vitrum` would.
BLOCK_OPTIONAL_EXTRAS = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _BlockOptionalExtras(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("atomate2", "mp_api"):
                raise ImportError(f"No module named {name!r} (simulated clean install)")

    sys.meta_path.insert(0, _BlockOptionalExtras())
    """
)


def run_without_extras(body):
    """Run `body` in a subprocess where atomate2 and mp_api cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_OPTIONAL_EXTRAS + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_package_imports_without_optional_extras():
    """`pip install vitrum && python -c "import vitrum"` must work.

    Every optional dependency has to stay behind a deferred import; one at module scope
    anywhere in the chain makes the package unimportable on a clean install, which in CI
    shows up as the whole suite failing at collection.
    """
    result = run_without_extras("import vitrum; print(vitrum.__version__)")
    assert result.returncode == 0, f"import vitrum failed:\n{result.stderr}"


def test_random_packing_works_without_optional_extras():
    """The default path through get_random_packed must not need an extra or an API key."""
    result = run_without_extras(
        """
        from vitrum.packing import get_random_packed
        atoms = get_random_packed("SiO2", target_atoms=24)
        assert len(atoms) == 24, len(atoms)
        assert atoms.get_volume() > 0
        """
    )
    assert result.returncode == 0, f"get_random_packed failed:\n{result.stderr}"


def test_database_backends_report_the_missing_extra():
    """A missing extra must name the extra, not surface a bare ModuleNotFoundError."""
    result = run_without_extras(
        """
        from vitrum.volume_estimation import get_volume
        try:
            get_volume("SiO2", {"Si": 1, "O": 2}, vol_per_atom_source="mp")
        except ImportError as e:
            assert "vitrum[volume_estimation]" in str(e), str(e)
        else:
            raise AssertionError("expected ImportError")
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "formula, structure, density",
    [
        ("SiO2", {"Si": 1, "O": 2}, 2.20),
        ("Na2SiO3", {"Na": 2, "Si": 1, "O": 3}, 2.50),
        ("CaSiO3", {"Ca": 1, "Si": 1, "O": 3}, 2.90),
    ],
)
def test_ionic_radius_estimate_is_within_calibration(formula, structure, density):
    """The default estimator must stay near the volume implied by the real density.

    The packing fraction in vitrum.volume_estimation is calibrated over a set of oxides with a spread
    of roughly -20%/+25% in volume; this guards against that calibration silently drifting.
    """
    from vitrum.volume_estimation import get_volume

    estimated = get_volume(formula, structure, vol_per_atom_source="ionic_radius")
    actual = get_volume(formula, structure, density=density)
    assert 0.7 < estimated / actual < 1.35, f"{formula}: {estimated / actual:.2f}x"


@pytest.mark.parametrize("scale", [1.2, 0.9])
def test_apply_strain_scales_the_volume_by_the_determinant(scale):
    """A deformation matrix D takes the cell volume to det(D) times itself.

    An isotropic scaling of s in each direction is s^3 of the volume, which is the check
    an EOS fit depends on: the deformations must actually reach the structure.
    """
    from pymatgen.core import Lattice, Structure

    from vitrum.packing import apply_strain_to_structure

    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    deformation = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]]

    (transformed,) = apply_strain_to_structure(structure, [deformation])
    assert transformed.final_structure.volume == pytest.approx(structure.volume * scale**3)
    # The input is a template for every deformation, so it must come back untouched.
    assert structure.volume == pytest.approx(64.0)


def test_apply_strain_returns_one_structure_per_deformation():
    """The deformations are applied independently, not composed onto one another."""
    from pymatgen.core import Lattice, Structure

    from vitrum.packing import apply_strain_to_structure

    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    deformations = [np.diag([s, s, s]).tolist() for s in (1.1, 1.0, 0.9)]

    transformed = apply_strain_to_structure(structure, deformations)
    assert len(transformed) == 3
    volumes = [t.final_structure.volume for t in transformed]
    assert volumes == pytest.approx([64.0 * s**3 for s in (1.1, 1.0, 0.9)])
