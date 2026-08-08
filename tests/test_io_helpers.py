"""Tests for the pure conversions in vitrum.io_helpers.

`formula_unit` is the fiddliest function in the package -- a regex tokeniser over exact
`Fraction` arithmetic, then a least-common-denominator reduction -- and every composition
that goes through `GlassGenerator` passes through it.
"""

import pytest
from conftest import N_NA, N_O, N_SI

from vitrum.io_helpers import (
    formula_unit,
    mass_density_to_number_density,
    number_density_to_mass_density,
)


def test_formula_unit_matches_the_documented_example():
    """The docstring's own example, which is the contract users read."""
    formula = "60.2SiO2-16.0B2O3-12.6Na2O-3.8Al2O3-5.7CaO-1.7ZrO2"
    assert formula_unit(formula) == "Si0.602B0.32Na0.252Al0.076Ca0.057Zr0.017O2.015"
    assert formula_unit(formula, integers=True) == "Si602B320Na252Al76Ca57Zr17O2015"


def test_formula_unit_reproduces_the_example_trajectory_composition():
    """70SiO2-30Na2O is the melt in examples/analysis; its atom counts are the answer.

    Si700 Na600 O1700 in the dump, and the smallest whole-number formula unit of the same
    composition is Si7Na6O17. Two independent things agreeing: a parser here, and what
    was actually simulated.
    """
    assert formula_unit("70SiO2-30Na2O") == "Si0.7Na0.6O1.7"
    assert formula_unit("70SiO2-30Na2O", integers=True) == "Si7Na6O17"
    assert (N_SI, N_NA, N_O) == (7 * 100, 6 * 100, 17 * 100)


def test_prefixes_are_proportions_and_need_not_sum_to_a_hundred():
    """The result is normalised, so doubling every prefix cannot change it."""
    assert formula_unit("60SiO2-40Na2O") == formula_unit("120SiO2-80Na2O")
    assert formula_unit("3SiO2-2Na2O") == formula_unit("60SiO2-40Na2O")


def test_components_may_be_separated_by_whitespace():
    assert formula_unit("70SiO2 30Na2O") == formula_unit("70SiO2-30Na2O")


def test_a_lone_oxide_is_its_own_formula_unit():
    """No prefix means a proportion of 1, and one component is the whole of it."""
    assert formula_unit("SiO2") == "SiO2"
    assert formula_unit("100SiO2", integers=True) == "SiO2"


def test_anions_last_puts_oxygen_at_the_end():
    """Glass formulas are written cations-first; the flag has to actually reorder them."""
    assert formula_unit("50Na2O-50SiO2").endswith("O1.5")
    assert formula_unit("50Na2O-50SiO2", anions_last=False).startswith("Na")


@pytest.mark.parametrize("formula", ["Si(OH)2", "xSiO2", "", "   "])
def test_unparseable_formulas_are_rejected(formula):
    """Parentheses are documented as unsupported, so they must say so rather than
    silently drop a group. An empty formula used to come back as an empty string.
    """
    with pytest.raises(ValueError, match="Cannot parse formula"):
        formula_unit(formula)


def test_a_trailing_separator_is_not_an_error():
    """Splitting drops empty components, so a stray '-' just names nothing."""
    assert formula_unit("70SiO2-30Na2O-") == formula_unit("70SiO2-30Na2O")


@pytest.mark.parametrize(
    "composition, density",
    [("SiO2", 2.2), ("Na2O", 2.27), ("Si7Na6O17", 2.47)],
)
def test_the_density_conversions_invert_each_other(composition, density):
    number_density = mass_density_to_number_density(composition, density)
    assert number_density_to_mass_density(composition, number_density) == pytest.approx(density)


def test_number_density_of_silica_is_the_known_value():
    """Vitreous SiO2 at 2.2 g/cm^3 holds 0.0662 atoms per cubic Angstrom.

    n = 3 * rho * N_A / M with M = 60.08 g/mol, computed by hand rather than read back
    from this function.
    """
    assert mass_density_to_number_density("SiO2", 2.2) == pytest.approx(0.0662, abs=1e-4)
