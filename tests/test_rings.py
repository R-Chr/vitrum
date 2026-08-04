"""Tests for vitrum.rings against crystalline silicon's known ring statistics."""

import pytest

from vitrum.rings import find_rings


def test_find_rings_accepts_bonds_none(silicon_small):
    """bonds=None must allow all bonds, as the docstring promises.

    None has to be guarded for in both places it is consumed: trimming the radii and
    filtering the bond list.
    """
    rings = find_rings(silicon_small, bonds=None, limit=8)
    assert len(rings) > 0


def test_bonds_none_matches_explicit_bonds(silicon_small):
    """For a single-species structure, bonds=None and the explicit list agree."""
    from_none = find_rings(silicon_small, bonds=None, limit=8)
    from_explicit = find_rings(silicon_small, bonds=[("Si", "Si")], limit=8)
    assert sorted(sorted(r) for r in from_none) == sorted(sorted(r) for r in from_explicit)


def test_silicon_rings_are_six_membered(silicon_small):
    """The shortest rings in the diamond structure are all 6-membered."""
    rings = find_rings(silicon_small, bonds=None, limit=8)
    assert {len(ring) for ring in rings} == {6}


@pytest.mark.parametrize("criterion", ["guttman", "king", "primitive"])
def test_all_criteria_give_six_membered_rings(silicon_small, criterion):
    rings = find_rings(silicon_small, bonds=None, limit=8, criterion=criterion)
    assert len(rings) > 0
    assert {len(ring) for ring in rings} == {6}


def test_unknown_criterion_raises(silicon_small):
    with pytest.raises(ValueError, match="Unknown ring criterion"):
        find_rings(silicon_small, bonds=None, criterion="nonsense")


def test_wrapping_rings_warn_once(monkeypatch):
    """Rejected rings must produce one aggregated warning, not one per ring.

    A real cell rejects thousands of candidates, so the report has to be summarised and
    it has to be a warning rather than a print, to stay filterable.
    """
    from ase.build import bulk

    from vitrum.rings import find_rings

    # A single cubic cell is small enough that some 6-rings wrap through the boundary.
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)

    with pytest.warns(UserWarning, match="wrap around the periodic cell") as record:
        find_rings(atoms, criterion="guttman")

    assert len(record) == 1
