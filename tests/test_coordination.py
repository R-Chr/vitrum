"""Tests for vitrum.coordination against crystalline silicon's known coordination."""

import numpy as np
import pytest
from ase import Atoms

from vitrum.coordination import Coordination

# Between the 1st (2.3517 A) and 2nd (3.840 A) neighbour shells in diamond Si.
SI_FIRST_SHELL_CUTOFF = 3.0
# Between the 1st (2.36 A) and 2nd (2.73 A) shells in fluorite CaF2.
CAF2_FIRST_SHELL_CUTOFF = 2.5


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
    with pytest.raises(ValueError, match="not present in the structure"):
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


def test_get_angles_requires_exactly_two_neighbour_types(silicon_small):
    """An angle spans two neighbours, so a third species cannot be silently dropped."""
    coordination = Coordination([silicon_small])
    with pytest.raises(ValueError, match="exactly two"):
        coordination.get_angles("Si", ["Si", "Si", "Si"], cutoff=SI_FIRST_SHELL_CUTOFF)


def test_get_angles_returns_one_array_per_frame(silicon_small):
    """The default grouping is one flat array per frame, matching the other raw accessors."""
    coordination = Coordination([silicon_small, silicon_small])
    angles = coordination.get_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    assert len(angles) == 2
    # Every Si has 4 neighbours, hence C(4, 2) = 6 angles each.
    assert all(len(frame) == 6 * len(silicon_small) for frame in angles)


def test_get_angles_per_atom_groups_by_centre(silicon_small):
    coordination = Coordination([silicon_small, silicon_small])
    per_frame = coordination.get_angles(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, per_atom=True
    )
    assert len(per_frame) == 2
    assert all(len(frame) == len(silicon_small) for frame in per_frame)
    assert all(len(entry) == 6 for frame in per_frame for entry in frame)


def test_get_angles_per_atom_flattens_to_the_default(silicon_small):
    """The two groupings must hold exactly the same angles, in the same order."""
    coordination = Coordination([silicon_small])
    flat = coordination.get_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    grouped = coordination.get_angles(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, per_atom=True
    )
    np.testing.assert_array_equal(flat[0], np.hstack(grouped[0]))


def test_get_angles_ordering_is_deterministic(silicon_small):
    """Pairs were once deduplicated through a set, leaving the order to hash layout."""
    coordination = Coordination([silicon_small])
    first = coordination.get_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)[0]
    second = coordination.get_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)[0]
    np.testing.assert_array_equal(first, second)


def test_coordination_number_per_atom_matches_known_value(silicon_small):
    coordination = Coordination([silicon_small])
    per_frame = coordination.get_coordination_numbers(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF, per_atom=True
    )
    assert len(per_frame) == 1
    assert set(per_frame[0].tolist()) == {4}


def test_coordination_number_per_atom_resolves_fluorite_asymmetry(fluorite_caf2):
    """Ca has 8 F neighbours but F has only 4 Ca: the centre/neighbour order matters."""
    coordination = Coordination([fluorite_caf2])
    ca = coordination.get_coordination_numbers(
        "Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF, per_atom=True
    )
    f = coordination.get_coordination_numbers(
        "F", "Ca", cutoff=CAF2_FIRST_SHELL_CUTOFF, per_atom=True
    )
    assert set(ca[0].tolist()) == {8}
    assert set(f[0].tolist()) == {4}


def test_coordination_per_atom_and_distribution_agree(fluorite_caf2):
    """The aggregated dict must be the binned form of the per-atom counts."""
    coordination = Coordination([fluorite_caf2, fluorite_caf2])
    per_frame = coordination.get_coordination_numbers(
        "Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF, per_atom=True
    )
    distribution = coordination.get_coordination_numbers(
        "Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF
    )
    counts = np.concatenate(per_frame)
    expected = {
        int(n): float((counts == n).sum() / counts.size) for n in np.unique(counts)
    }
    assert distribution == pytest.approx(expected)


def test_get_neighbors_returns_dict_keyed_by_species(silicon_small):
    """get_neighbors returns a dict of per-species neighbour lists, not a list of counts."""
    coordination = Coordination([silicon_small])
    per_frame = coordination.get_neighbors("Si", SI_FIRST_SHELL_CUTOFF)
    assert len(per_frame) == 1
    neighbors = per_frame[0]
    assert isinstance(neighbors, dict)
    assert set(neighbors) == {"Si"}
    assert len(neighbors["Si"]) == len(silicon_small)
    assert all(len(entry) == 4 for entry in neighbors["Si"])


def test_get_neighbors_returns_global_atom_indices(fluorite_caf2):
    """Entries must index straight into the frame, not into that species' own atoms."""
    neighbors = Coordination([fluorite_caf2]).get_neighbors(
        "Ca", CAF2_FIRST_SHELL_CUTOFF
    )[0]
    symbols = np.array(fluorite_caf2.get_chemical_symbols())
    for species, per_center in neighbors.items():
        for entry in per_center:
            assert set(symbols[entry].tolist()) <= {species}
    # A species-relative F index would run 0..n_F-1 and so never reach the last atom.
    assert max(entry.max() for entry in neighbors["F"] if entry.size) >= len(
        np.where(symbols == "F")[0]
    )


def test_get_neighbors_auto_cutoff_matches_an_explicit_one(silicon_diamond):
    coordination = Coordination([silicon_diamond])
    auto = coordination.get_neighbors("Si", "Auto")[0]
    explicit = coordination.get_neighbors("Si", SI_FIRST_SHELL_CUTOFF)[0]
    assert set(auto) == set(explicit)
    for species in auto:
        for a, b in zip(auto[species], explicit[species]):
            np.testing.assert_array_equal(a, b)


def test_get_neighbors_rejects_bad_cutoff_type(silicon_small):
    with pytest.raises(TypeError):
        Coordination([silicon_small]).get_neighbors("Si", object())


def test_get_neighbors_rejects_dict_missing_a_species(silicon_small):
    with pytest.raises(KeyError, match="No cutoff defined"):
        Coordination([silicon_small]).get_neighbors("Si", {"Ge": 3.0})


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.get_neighbors("Ge", 3.0),
        lambda c: c.get_angles("Ge", "Si"),
        lambda c: c.get_coordination_numbers("Ge", "Si"),
        lambda c: c.get_bridging_analysis("Ge", "Si"),
    ],
)
def test_absent_species_is_rejected_consistently(silicon_small, call):
    """All four species-taking methods must reject an absent species the same way.

    An absent species otherwise surfaces as an all-zero PDF with no first minimum, so the
    user is told the automatic cutoff search failed rather than that the species is missing.
    """
    with pytest.raises(ValueError, match="not present in the structure"):
        call(Coordination([silicon_small]))


def _q_species_chain():
    """Three corner-sharing SiO4 tetrahedra in a row, with known Q speciation.

    Built as a linear chain Si-O-Si-O-Si with terminal oxygens on each end, in a cell big
    enough that nothing bonds through the boundary. Only the two chain-internal oxygens
    are bonded to two Si, so the end tetrahedra are Q1 and the middle one is Q2.
    """
    d = 1.6  # Si-O bond length, comfortably under the 2.0 A cutoff used below.
    positions, symbols = [], []
    # Si at x = 0, 2d, 4d; bridging O between them; one terminal O on each end.
    for i in range(3):
        positions.append([2 * d * i, 0.0, 0.0])
        symbols.append("Si")
    for x in (d, 3 * d):  # bridging oxygens
        positions.append([x, 0.0, 0.0])
        symbols.append("O")
    for x in (-d, 4 * d + d):  # terminal oxygens
        positions.append([x, 0.0, 0.0])
        symbols.append("O")
    return Atoms(symbols, positions=positions, cell=[40.0] * 3, pbc=True)


def test_bridging_analysis_counts_known_q_species():
    """Q1-Q2-Q1 chain: two thirds Q1, one third Q2."""
    coordination = Coordination([_q_species_chain()])
    distribution = coordination.get_bridging_analysis("Si", "O", cutoff=2.0)
    assert distribution == {1: pytest.approx(2 / 3), 2: pytest.approx(1 / 3)}


def test_bridging_analysis_per_atom_gives_the_raw_counts():
    coordination = Coordination([_q_species_chain()])
    per_frame = coordination.get_bridging_analysis("Si", "O", cutoff=2.0, per_atom=True)
    assert len(per_frame) == 1
    # Si atoms are in build order: the two chain ends flank the middle one.
    assert per_frame[0].tolist() == [1, 2, 1]


def test_bridging_analysis_aggregates_over_frames():
    """Two identical frames give the same fractions as one, with twice the atoms."""
    chain = _q_species_chain()
    one = Coordination([chain]).get_bridging_analysis("Si", "O", cutoff=2.0)
    two = Coordination([chain, chain]).get_bridging_analysis("Si", "O", cutoff=2.0)
    assert one == pytest.approx(two)


def test_bridging_analysis_former_types_restricts_what_bridges():
    """With no Si counted as a former, no oxygen can bridge, so everything is Q0."""
    atoms = _q_species_chain()
    # Naming only O as a former means no bridging oxygen has two formers around it.
    distribution = Coordination([atoms]).get_bridging_analysis(
        "Si", "O", former_types=["O"], cutoff=2.0
    )
    assert distribution == {0: pytest.approx(1.0)}


def test_bridging_analysis_rejects_non_list_former_types():
    with pytest.raises(TypeError, match="former_types"):
        Coordination([_q_species_chain()]).get_bridging_analysis(
            "Si", "O", former_types=("Si",), cutoff=2.0
        )


def test_bridging_analysis_rejects_a_bare_string_former_type():
    """A str is iterable, so the type check has to run before the species guard.

    Otherwise `former_types="Si"` is splatted into the guard as 'S' and 'i' and the user
    is told those species are missing, rather than that a list was expected.
    """
    with pytest.raises(TypeError, match="former_types"):
        Coordination([_q_species_chain()]).get_bridging_analysis(
            "Si", "O", former_types="Si", cutoff=2.0
        )


def test_coordination_accepts_plain_ase_atoms(silicon_small):
    """The class must not need, or produce, any vitrum-specific atoms type."""
    coordination = Coordination([silicon_small])
    assert type(coordination.atoms_list[0]) is Atoms
    assert coordination.atoms_list[0] is silicon_small


def test_get_bonds_returns_one_entry_per_frame(silicon_small):
    coordination = Coordination([silicon_small, silicon_small])
    bonds = coordination.get_bonds("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    assert len(bonds) == 2
    assert all(set(b.counts().tolist()) == {4} for b in bonds)


def test_get_bonds_is_what_the_other_methods_reduce(fluorite_caf2):
    """The public primitive must give the same answer as the methods derived from it."""
    coordination = Coordination([fluorite_caf2])
    bonds = coordination.get_bonds("Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF)[0]

    per_atom = coordination.get_coordination_numbers(
        "Ca", "F", cutoff=CAF2_FIRST_SHELL_CUTOFF, per_atom=True
    )[0]
    np.testing.assert_array_equal(bonds.counts(), per_atom)

    neighbors = coordination.get_neighbors("Ca", CAF2_FIRST_SHELL_CUTOFF)[0]
    for from_bonds, from_method in zip(bonds.lists(), neighbors["F"]):
        np.testing.assert_array_equal(from_bonds, from_method)


def test_get_bonds_auto_cutoff_matches_an_explicit_one(silicon_diamond):
    coordination = Coordination([silicon_diamond])
    auto = coordination.get_bonds("Si", "Si")[0]
    explicit = coordination.get_bonds("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)[0]
    np.testing.assert_array_equal(auto.counts(), explicit.counts())


def test_get_bonds_accepts_multi_species_selections():
    """A selection may name several species, which is how former_types is handled."""
    chain = _q_species_chain()
    bonds = Coordination([chain]).get_bonds(["Si", "O"], "O", cutoff=2.0)[0]
    symbols = np.array(chain.get_chemical_symbols())
    assert set(symbols[bonds.centers].tolist()) == {"Si", "O"}
    assert set(symbols[bonds.neighs].tolist()) == {"O"}


def test_get_bonds_rejects_an_absent_species(silicon_small):
    with pytest.raises(ValueError, match="not present in the structure"):
        Coordination([silicon_small]).get_bonds("Ge", "Si", cutoff=3.0)


def test_empty_atoms_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        Coordination([])


def test_a_single_atoms_object_is_treated_as_one_frame(silicon_small):
    """Atoms is sized and indexable, so an unwrapped frame would iterate into Atom objects."""
    coordination = Coordination(silicon_small)
    assert coordination.atoms_list == [silicon_small]
    assert coordination.get_coordination_numbers(
        "Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF
    ) == {4: pytest.approx(1.0)}


def test_non_atoms_frames_are_rejected(silicon_small):
    with pytest.raises(TypeError, match="must contain Atoms objects"):
        Coordination([silicon_small, "not a structure"])


def test_a_generator_of_frames_is_accepted(silicon_small):
    """list() consumes the generator, so frame 0 has to be read back from the list."""
    coordination = Coordination(atoms for atoms in (silicon_small, silicon_small))
    assert len(coordination.atoms_list) == 2
    assert coordination.species.tolist() == ["Si"]


@pytest.mark.parametrize(
    "method",
    ["get_angles", "get_coordination_numbers"],
)
def test_unknown_cutoff_string_is_rejected_by_every_method(silicon_small, method):
    """Only the exact string "Auto" is a cutoff; anything else once died inside NumPy."""
    coordination = Coordination([silicon_small])
    with pytest.raises(ValueError, match="Invalid cutoff"):
        getattr(coordination, method)("Si", "Si", cutoff="auto")


def test_boolean_cutoff_is_rejected(silicon_small):
    """True is an int, and would otherwise be silently read as a 1 Angstrom cutoff."""
    coordination = Coordination([silicon_small])
    with pytest.raises(TypeError, match="Invalid cutoff"):
        coordination.get_coordination_numbers("Si", "Si", cutoff=True)


def test_numpy_scalar_cutoff_is_accepted(silicon_small):
    """np.int64 is not an int, so a cutoff read out of an array must still work."""
    coordination = Coordination([silicon_small])
    distribution = coordination.get_coordination_numbers(
        "Si", "Si", cutoff=np.int64(3)
    )
    assert distribution == {4: pytest.approx(1.0)}


def test_angles_do_not_double_count_same_species_with_two_cutoffs():
    """Same-species arms at two different cutoffs must not double-count a qualifying pair.

    Regression: the product branch fired whenever the two arm neighbour lists differed,
    which happens for same-species arms as soon as the cutoffs differ, and it then counted
    both (x, y) and (y, x) for any pair inside the smaller cutoff -- the same physical angle
    twice.

    Three O atoms sit on the axes around a central X, at 1.0, 1.3 and 1.6 A. cutoff=[2.5,
    1.5] puts all three O within the first arm's cutoff but only the two closer ones within
    the second's, so exactly three unordered pairs qualify: {O1, O2} (in both arms), {O1,
    O3} and {O2, O3} (each in one arm only). The old code counted {O1, O2} twice.
    """
    atoms = Atoms(
        "XOOO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.6]],
        cell=[40.0] * 3,
        pbc=True,
    )
    coordination = Coordination([atoms])
    angles = coordination.get_angles("X", ["O", "O"], cutoff=[2.5, 1.5])
    assert len(angles[0]) == 3


def test_angles_same_species_equal_cutoffs_are_unaffected(silicon_small):
    """The same-species fix must not change the equal-cutoff case it used to handle."""
    coordination = Coordination([silicon_small])
    single = coordination.get_angles("Si", "Si", cutoff=SI_FIRST_SHELL_CUTOFF)
    two = coordination.get_angles(
        "Si", ["Si", "Si"], cutoff=[SI_FIRST_SHELL_CUTOFF, SI_FIRST_SHELL_CUTOFF]
    )
    np.testing.assert_array_equal(single[0], two[0])


def test_angle_distribution_with_no_angles_raises(silicon_small):
    """A zero-count density histogram divides by zero and returns NaN; say so instead."""
    coordination = Coordination([silicon_small])
    with pytest.raises(ValueError, match="No Si-Si-Si angles found"):
        coordination.get_angle_distribution("Si", "Si", cutoff=0.5, range=(0.0, 180.0))
