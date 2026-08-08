"""Tests for vitrum.structure_gen: composition sampling, not structure packing.

`get_structures` goes through `get_random_packed`, which test_packaging.py already covers;
what is checked here is that the sampled mole fractions are a valid composition and survive
the conversion to formulas and pymatgen objects.
"""

import pytest

from vitrum.structure_gen import Compositions, GlassGenerator

UNITS = ["SiO2", "Na2O", "CaO"]
META_COLUMNS = {"n_components", "former_sum"}


@pytest.fixture
def generator():
    return GlassGenerator(units=UNITS, network_formers={"SiO2"}, seed=0)


def value_columns(compositions):
    return [c for c in compositions.df.columns if c not in META_COLUMNS]


def test_sample_returns_the_requested_number(generator):
    compositions = generator.sample(scheme="sobol", n=8)
    assert isinstance(compositions, Compositions)
    assert len(compositions) == 8
    assert compositions.mode == "unit"


def test_every_row_is_a_valid_mole_fraction(generator):
    """Mole fractions are non-negative and sum to 1; anything else is not a composition.

    The sampled fractions are rounded to four decimals for readability, so a row can land
    on 0.9999; the tolerance is that rounding and no more.
    """
    compositions = generator.sample(scheme="sobol", n=16)
    values = compositions.df[value_columns(compositions)]
    assert (values >= 0).all().all()
    assert values.sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-3)


def test_the_seed_makes_sampling_reproducible():
    """Two generators with the same seed must sample the same compositions."""
    first = GlassGenerator(units=UNITS, seed=1).sample(n=8)
    second = GlassGenerator(units=UNITS, seed=1).sample(n=8)
    assert first.df.equals(second.df)


def test_different_seeds_sample_differently():
    first = GlassGenerator(units=UNITS, seed=1).sample(n=8)
    second = GlassGenerator(units=UNITS, seed=2).sample(n=8)
    assert not first.df.equals(second.df)


def test_x_min_is_respected_by_the_active_components(generator):
    """A component is either absent or present in at least x_min; nothing in between.

    Trace amounts are not glasses anyone makes, which is what the floor is for.
    """
    generator = GlassGenerator(units=UNITS, x_min=0.1, seed=3)
    values = generator.sample(scheme="sobol", n=24)
    frame = values.df[value_columns(values)]
    active = frame.values[frame.values > 0]
    assert active.min() >= 0.1


def test_min_former_sum_is_respected():
    """The constraint has to bind, not just be accepted."""
    generator = GlassGenerator(
        units=UNITS, network_formers={"SiO2"}, min_former_sum=0.5, seed=4
    )
    compositions = generator.sample(scheme="sobol", n=24)
    assert (compositions.df["former_sum"] >= 0.5).all()


def test_to_formulas_conserves_the_sampled_elements(generator):
    """Every element of an active unit must appear in the formula it is converted to."""
    compositions = generator.sample(scheme="sobol", n=8)
    formulas = compositions.to_formulas()
    assert len(formulas) == len(compositions)
    for formula, (_, row) in zip(formulas, compositions.df.iterrows()):
        if row["SiO2"] > 0:
            assert "Si" in formula
        if row["Na2O"] > 0:
            assert "Na" in formula
        assert "O" in formula


def test_to_pymatgen_gives_whole_number_compositions(generator):
    """`get_random_packed` needs integer counts, which is what the reduction is for."""
    compositions = generator.sample(scheme="sobol", n=8)
    for composition in compositions.to_pymatgen():
        for amount in composition.as_dict().values():
            assert amount == pytest.approx(round(amount))


def test_to_pymatgen_and_to_formulas_describe_the_same_composition(generator):
    """Two renderings of one row must agree on the element ratios.

    Not bit-for-bit: the pymatgen route reduces to the smallest whole-number formula and
    the string route rounds to six significant figures, so they land a rounding apart.
    """
    from pymatgen.core import Composition

    compositions = generator.sample(scheme="sobol", n=6)
    for composition, formula in zip(compositions.to_pymatgen(), compositions.to_formulas()):
        from_object = composition.fractional_composition
        from_string = Composition(formula).fractional_composition
        assert set(from_object) == set(from_string)
        for element in from_object:
            assert from_object[element] == pytest.approx(from_string[element], abs=1e-4)


def test_elemental_mode_sets_its_own_mode():
    """Elemental mode charge-balances rather than mixing units, and has its own schemes."""
    generator = GlassGenerator(
        elements={"formers": ["Si"], "modifiers": ["Na"], "anions": ["O"]}, seed=0
    )
    compositions = generator.sample(scheme="random", n=6)
    assert compositions.mode == "elemental"
    assert len(compositions) == 6
    fractions = compositions.df[value_columns(compositions)].sum(axis=1).to_numpy()
    assert fractions == pytest.approx(1.0, abs=1e-3)


def test_a_unit_mode_scheme_is_rejected_in_elemental_mode():
    """The two modes have different scheme sets; naming the wrong one must say so."""
    generator = GlassGenerator(elements={"formers": ["Si"], "anions": ["O"]}, seed=0)
    with pytest.raises(ValueError, match="Unknown elemental-mode scheme"):
        generator.sample(scheme="sobol", n=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"units": UNITS, "elements": {"formers": ["Si"]}},
    ],
)
def test_exactly_one_of_units_or_elements_is_required(kwargs):
    """Neither and both are equally ambiguous, so both must be refused."""
    with pytest.raises(ValueError, match="exactly one"):
        GlassGenerator(**kwargs)
