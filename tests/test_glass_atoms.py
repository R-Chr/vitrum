"""Tests for the deprecated GlassAtoms shim.

The numerical contracts themselves live in test_geometry.py and test_coordination.py, next
to the code that now owns them. What is checked here is that the shim still exists, still
returns what it always returned, and says so loudly. Deleting this file is part of removing
GlassAtoms in 2.0.0.
"""

import numpy as np
import pytest
from ase import Atoms

from vitrum.coordination import Coordination
from vitrum.geometry import distance_matrix
from vitrum.glass_atoms import GlassAtoms
from vitrum.io_helpers import get_density

SI_FIRST_SHELL_CUTOFF = 3.0


@pytest.mark.parametrize(
    "call, replacement",
    [
        (lambda a: a.get_dist(), "geometry.distance_matrix"),
        (lambda a: a.get_pdf(["Si", "Si"]), "get_partial_pdf"),
        (lambda a: a.get_all_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF), "get_angles"),
        (
            lambda a: a.get_coordination_number("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF),
            "get_coordination_numbers",
        ),
        (
            lambda a: a.get_bridging_analysis("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF),
            "get_bridging_analysis",
        ),
        (lambda a: a.get_density(), "io_helpers.get_density"),
        (lambda a: a.get_neighbors("Si", SI_FIRST_SHELL_CUTOFF), "get_neighbors"),
        (lambda a: a.set_new_chemical_symbols({14: "Si"}), "correct_atom_types"),
    ],
)
def test_every_method_warns_and_names_its_replacement(silicon_small, call, replacement):
    """A deprecation warning is only useful if it says what to use instead."""
    atoms = GlassAtoms(silicon_small)
    with pytest.warns(DeprecationWarning, match="removed in vitrum 2.0.0"):
        call(atoms)
    with pytest.warns(DeprecationWarning, match=replacement):
        call(atoms)


def test_get_dist_still_matches_distance_matrix(silicon_small):
    atoms = GlassAtoms(silicon_small)
    with pytest.deprecated_call():
        np.testing.assert_allclose(atoms.get_dist(), distance_matrix(silicon_small), atol=1e-12)


def test_get_density_still_matches_io_helpers(silicon_small):
    atoms = GlassAtoms(silicon_small)
    with pytest.deprecated_call():
        assert atoms.get_density() == pytest.approx(get_density(silicon_small))


def test_get_coordination_number_still_matches_coordination(silicon_small):
    atoms = GlassAtoms(silicon_small)
    with pytest.deprecated_call():
        shim = atoms.get_coordination_number("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    replacement = Coordination([silicon_small]).get_coordination_numbers(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, per_atom=True
    )
    assert shim == replacement[0].tolist()


def test_get_bridging_analysis_still_matches_coordination(silicon_small):
    atoms = GlassAtoms(silicon_small)
    with pytest.deprecated_call():
        shim = atoms.get_bridging_analysis("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    replacement = Coordination([silicon_small]).get_bridging_analysis(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, per_atom=True
    )
    assert shim == replacement[0].tolist()


def test_get_neighbors_keeps_returning_species_relative_indices(silicon_small):
    """Coordination now returns global indices; the shim must not follow it.

    Both index the same atoms, so mapping the shim's entries back through that species'
    atom list has to reproduce the class's output exactly.
    """
    atoms = GlassAtoms(silicon_small)
    with pytest.deprecated_call():
        shim = atoms.get_neighbors("Si", SI_FIRST_SHELL_CUTOFF)
    replacement = Coordination([silicon_small]).get_neighbors("Si", SI_FIRST_SHELL_CUTOFF)[0]
    symbols = np.array(silicon_small.get_chemical_symbols())

    assert set(shim) == set(replacement)
    for species in shim:
        species_atoms = np.where(symbols == species)[0]
        for relative, glob in zip(shim[species], replacement[species]):
            np.testing.assert_array_equal(species_atoms[relative], glob)


def test_get_pdf_indicies_override_is_preserved(silicon_small):
    """The misspelt `indicies=` keyword is part of the shipped API; it must keep working."""
    atoms = GlassAtoms(silicon_small)
    all_si = np.arange(len(silicon_small))
    with pytest.deprecated_call():
        _, by_index = atoms.get_pdf(["Si", "Si"], indicies=[all_si, all_si])
    with pytest.deprecated_call():
        _, by_symbol = atoms.get_pdf(["Si", "Si"])
    np.testing.assert_allclose(by_index, by_symbol)


def test_get_pdf_rejects_non_symbol_non_integer_targets(silicon_small):
    atoms = GlassAtoms(silicon_small)
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="strings or integers"):
        atoms.get_pdf([1.5, 1.5])


def test_set_new_chemical_symbols_still_renames_in_place():
    atoms = GlassAtoms(Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10] * 3, pbc=True))
    with pytest.deprecated_call():
        atoms.set_new_chemical_symbols({1: "O"})
    assert atoms.get_chemical_symbols() == ["O", "O"]


def test_glass_atoms_is_still_a_drop_in_ase_atoms(silicon_small):
    """Existing scripts pass GlassAtoms straight into ASE and into the analysis classes."""
    atoms = GlassAtoms(silicon_small)
    assert isinstance(atoms, Atoms)
    numbers = Coordination([atoms]).get_coordination_numbers(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF
    )
    assert numbers == {4: pytest.approx(1.0)}
