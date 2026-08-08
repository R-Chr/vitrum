"""Tests for the cutoff grammar shared by every Coordination method.

A cutoff belongs to a bond, not to an atom, so the same spellings have to mean the same
thing wherever they are accepted. What is checked here is agreement between the spellings
-- a dict, a list and a number that describe the same bonds must give the same numbers --
and that the one method without a neighbour-type argument refuses a spelling it cannot
order.
"""

import numpy as np
import pytest
from ase import Atoms
from conftest import CAF2_FIRST_SHELL_CUTOFF, SI_FIRST_SHELL_CUTOFF

from vitrum.coordination import Coordination


@pytest.fixture
def borosilicate():
    """Si-O ... B, where the O is bonded to the Si but merely near the B.

    The Si-O separation is 1.6 A, a real bond; the B-O separation is 1.9 A, well beyond the
    ~1.4 A of a real B-O bond. A single cutoff generous enough for Si-O therefore invents a
    B-O bond and reports the oxygen as bridging, while a per-bond cutoff judges the boron
    at its own bond length and does not.
    """
    positions = [
        [0.0, 0.0, 0.0],  # Si
        [1.6, 0.0, 0.0],  # the O in question: 1.6 from Si, 1.9 from B
        [3.5, 0.0, 0.0],  # B
        [-1.6, 0.0, 0.0],  # terminal O on Si
        [4.9, 0.0, 0.0],  # terminal O on B, at a real 1.4 A B-O bond length
    ]
    return Atoms("SiOBOO", positions=positions, cell=[40.0] * 3, pbc=True)


@pytest.fixture
def caf2(fluorite_caf2):
    return Coordination([fluorite_caf2])


# --- the spellings agree with one another -------------------------------------------------


def test_number_and_dict_agree_for_coordination_numbers(caf2):
    """One cutoff written three ways must count the same bonds."""
    number = caf2.get_coordination_numbers("Ca", ["F", "Ca"], CAF2_FIRST_SHELL_CUTOFF)
    by_species = caf2.get_coordination_numbers(
        "Ca", ["F", "Ca"], {"F": CAF2_FIRST_SHELL_CUTOFF, "Ca": CAF2_FIRST_SHELL_CUTOFF}
    )
    by_bond = caf2.get_coordination_numbers(
        "Ca",
        ["F", "Ca"],
        {("Ca", "F"): CAF2_FIRST_SHELL_CUTOFF, ("Ca", "Ca"): CAF2_FIRST_SHELL_CUTOFF},
    )
    assert number == by_species == by_bond


def test_dict_distinguishes_the_two_bonds(caf2):
    """A per-bond dict must actually apply different cutoffs, not just accept them."""
    both_short = caf2.get_coordination_numbers("Ca", ["F", "Ca"], {"F": 2.5, "Ca": 2.5})
    ca_longer = caf2.get_coordination_numbers("Ca", ["F", "Ca"], {"F": 2.5, "Ca": 4.0})
    assert both_short != ca_longer


def test_bond_key_order_does_not_matter(caf2):
    """A cutoff is a property of an unordered bond, so ("F", "Ca") is the ("Ca", "F") one."""
    forward = caf2.get_coordination_numbers("Ca", "F", {("Ca", "F"): 2.5})
    reversed_key = caf2.get_coordination_numbers("Ca", "F", {("F", "Ca"): 2.5})
    assert forward == reversed_key


def test_bond_key_beats_the_species_shorthand(caf2):
    """The explicit spelling has to win, or a shared dict could not be refined per bond."""
    refined = caf2.get_coordination_numbers("Ca", "F", {"F": 4.0, ("Ca", "F"): 2.5})
    assert refined == caf2.get_coordination_numbers("Ca", "F", 2.5)


def test_one_dict_serves_every_method(fluorite_caf2):
    """The point of keying on bonds: the same dict is reusable across the whole class."""
    coordination = Coordination([fluorite_caf2])
    cutoffs = {"F": CAF2_FIRST_SHELL_CUTOFF, "Ca": CAF2_FIRST_SHELL_CUTOFF}

    assert coordination.get_coordination_numbers("Ca", "F", cutoffs) == (
        coordination.get_coordination_numbers("Ca", "F", CAF2_FIRST_SHELL_CUTOFF)
    )
    assert coordination.get_bridging_analysis("Ca", "F", cutoff=cutoffs) == (
        coordination.get_bridging_analysis("Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF)
    )
    np.testing.assert_array_equal(
        coordination.get_angles("Ca", "F", cutoffs)[0],
        coordination.get_angles("Ca", "F", CAF2_FIRST_SHELL_CUTOFF)[0],
    )
    assert len(coordination.get_bonds("Ca", "F", cutoffs)[0]) == len(
        coordination.get_bonds("Ca", "F", CAF2_FIRST_SHELL_CUTOFF)[0]
    )
    from_dict = coordination.get_neighbors("Ca", cutoffs)[0]
    from_number = coordination.get_neighbors("Ca", CAF2_FIRST_SHELL_CUTOFF)[0]
    for species in from_number:
        for a, b in zip(from_dict[species], from_number[species]):
            np.testing.assert_array_equal(a, b)


def test_extra_dict_keys_are_ignored(caf2):
    """A shared dict names bonds a given call does not measure; that must not be an error."""
    assert caf2.get_coordination_numbers(
        "Ca", "F", {"F": 2.5, "Ca": 2.5, ("Na", "O"): 9.9}
    ) == caf2.get_coordination_numbers("Ca", "F", 2.5)


# --- per-former cutoffs in the bridging analysis ------------------------------------------


def test_bridging_analysis_judges_each_former_at_its_own_cutoff(borosilicate):
    """A per-bond dict has to stop a generous Si-O cutoff from inventing a B-O bond.

    At a shared 2.0 A both formers reach the oxygen, so it counts as bridging and Si is Q1.
    Judging boron at its own 1.6 A leaves the 1.9 A B-O separation out, so the oxygen has
    one former and Si is Q0 -- which a single shared cutoff cannot express.
    """
    coordination = Coordination([borosilicate])
    formers = ["Si", "B"]

    shared = coordination.get_bridging_analysis(
        "Si", "O", former_types=formers, cutoff=2.0, per_atom=True
    )
    assert shared[0].tolist() == [1]

    per_bond = coordination.get_bridging_analysis(
        "Si",
        "O",
        former_types=formers,
        cutoff={("Si", "O"): 2.0, ("B", "O"): 1.6},
        per_atom=True,
    )
    assert per_bond[0].tolist() == [0]


def test_bridging_analysis_species_key_still_applies_to_every_bond(borosilicate):
    """{"O": x} keys on the bridge, which every measured bond shares, so it means one cutoff."""
    coordination = Coordination([borosilicate])
    shared = coordination.get_bridging_analysis(
        "Si", "O", former_types=["Si", "B"], cutoff={"O": 2.0}, per_atom=True
    )
    number = coordination.get_bridging_analysis(
        "Si", "O", former_types=["Si", "B"], cutoff=2.0, per_atom=True
    )
    np.testing.assert_array_equal(shared[0], number[0])


def test_bridging_analysis_ignores_a_repeated_former(borosilicate):
    """Each former contributes its bonds once, so naming one twice must not double it."""
    coordination = Coordination([borosilicate])
    once = coordination.get_bridging_analysis(
        "Si", "O", former_types=["Si", "B"], cutoff=2.0, per_atom=True
    )
    twice = coordination.get_bridging_analysis(
        "Si", "O", former_types=["Si", "B", "B"], cutoff=2.0, per_atom=True
    )
    np.testing.assert_array_equal(once[0], twice[0])


# --- rejections ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda c, x: c.get_coordination_numbers("Si", "Si", x),
        lambda c, x: c.get_angles("Si", "Si", x),
        lambda c, x: c.get_bonds("Si", "Si", x),
        lambda c, x: c.get_bridging_analysis("Si", "Si", cutoff=x),
        lambda c, x: c.get_neighbors("Si", x),
    ],
)
def test_every_method_rejects_an_unusable_cutoff_the_same_way(silicon_small, call):
    coordination = Coordination([silicon_small])
    with pytest.raises(TypeError, match="Invalid cutoff"):
        call(coordination, object())


@pytest.mark.parametrize(
    "call",
    [
        lambda c, x: c.get_coordination_numbers("Si", "Si", x),
        lambda c, x: c.get_angles("Si", "Si", x),
        lambda c, x: c.get_bonds("Si", "Si", x),
        lambda c, x: c.get_bridging_analysis("Si", "Si", cutoff=x),
        lambda c, x: c.get_neighbors("Si", x),
    ],
)
def test_every_method_rejects_an_unknown_cutoff_string(silicon_small, call):
    coordination = Coordination([silicon_small])
    with pytest.raises(ValueError, match="Invalid cutoff"):
        call(coordination, "auto")


@pytest.mark.parametrize(
    "call",
    [
        lambda c, x: c.get_coordination_numbers("Si", "Si", x),
        lambda c, x: c.get_angles("Si", "Si", x),
        lambda c, x: c.get_bonds("Si", "Si", x),
        lambda c, x: c.get_bridging_analysis("Si", "Si", cutoff=x),
        lambda c, x: c.get_neighbors("Si", x),
    ],
)
def test_every_method_rejects_a_dict_missing_the_bond(silicon_small, call):
    coordination = Coordination([silicon_small])
    with pytest.raises(KeyError, match="No cutoff defined"):
        call(coordination, {"Ge": 3.0})


def test_get_neighbors_rejects_a_list(silicon_small):
    """A cutoff is named by the bond it belongs to; a bare list names nothing."""
    coordination = Coordination([silicon_small])
    with pytest.raises(TypeError, match="Invalid cutoff"):
        coordination.get_neighbors("Si", [SI_FIRST_SHELL_CUTOFF])


def test_a_key_matching_no_bond_is_reported_as_a_missing_cutoff(silicon_small):
    """A malformed or irrelevant key simply matches nothing, so the bond is left uncovered."""
    coordination = Coordination([silicon_small])
    with pytest.raises(KeyError, match="Si-Si bond needs an entry"):
        coordination.get_coordination_numbers("Si", "Si", {("Si", "O", "Na"): 2.0})


def test_auto_resolves_each_bond_once(silicon_small):
    """A bond named twice must not be resolved twice, and must resolve to one value."""
    coordination = Coordination([silicon_small])
    auto = coordination.get_bridging_analysis(
        "Si", "Si", former_types=["Si"], cutoff="Auto", per_atom=True
    )
    assert len(auto) == 1


def test_auto_resolves_on_a_triclinic_cell(triclinic_atoms):
    """Auto-resolution used to need an orthorhombic cell: it fell back to the KD-tree bond
    backend, which raised `NotImplementedError` off `require_orthorhombic` for a triclinic
    one. The bond backend is now `ase.neighborlist.neighbor_list`, which handles a triclinic
    cell directly, so "Auto" must resolve here without raising rather than pin an exact
    distribution -- the fixture shears the cell without moving the atoms, so its PDF is not
    a physically meaningful one to check numbers against.
    """
    coordination = Coordination([triclinic_atoms])
    distribution = coordination.get_coordination_numbers("Si", "Si", cutoff="Auto")
    assert distribution
    assert sum(distribution.values()) == pytest.approx(1.0)