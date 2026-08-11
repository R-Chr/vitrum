"""Tests against the example trajectory: a 30Na2O-70SiO2 melt, the material vitrum is for.

Every expected value here is fixed by the composition or by silicate chemistry, never by
what vitrum returns. Na600 O1700 Si700 is 70 SiO2 . 30 Na2O, and from that alone:

  - silicon is tetrahedral, so every Si has 4 oxygens and nothing else;
  - each Na needs one non-bridging oxygen to charge-balance it, so 600 of the 1700
    oxygens are non-bridging and the other 1100 bridge two tetrahedra;
  - the same 600 broken bridges are shared over 700 tetrahedra, so <Qn> = 4 - 600/700.

These are the quantities a crystal cannot express -- a distribution rather than a single
integer -- and they are what the package is used to measure.
"""

import numpy as np
import pytest
from conftest import (
    BRIDGING_OXYGEN_FRACTION,
    MEAN_Q_SPECIES,
    N_NA,
    N_O,
    N_SI,
    SI_O_CUTOFF,
)

from vitrum.coordination import Coordination
from vitrum.diffusion import Diffusion
from vitrum.rings import find_rings
from vitrum.scattering import Scattering
from vitrum.structure_validation import homogeneity_checker

# The Si-O bond length in a silicate, from the first peak of g_SiO(r).
SI_O_BOND_LENGTH = 1.60


def test_the_trajectory_is_the_composition_the_tests_assume(sodium_silicate_frame):
    """If the type-to-symbol map is ever wrong, every expectation below is meaningless."""
    symbols = sodium_silicate_frame.get_chemical_symbols()
    assert symbols.count("Na") == N_NA
    assert symbols.count("O") == N_O
    assert symbols.count("Si") == N_SI


def test_every_silicon_is_tetrahedral(sodium_silicate):
    """A silicate melt has no under- or over-coordinated Si to speak of: SiO4, all of it."""
    coordination = Coordination(sodium_silicate)
    distribution = coordination.get_coordination_numbers("Si", "O", cutoff=SI_O_CUTOFF)
    assert distribution[4] == pytest.approx(1.0, abs=0.005)


def test_bridging_oxygen_fraction_is_fixed_by_the_sodium_content(sodium_silicate):
    """Each Na breaks one bridge, so the bridging fraction is 1 - Na/O = 0.647.

    Read from the oxygen's side: an O bonded to two Si bridges, one bonded to a single Si
    does not. Nothing about this number comes from vitrum.
    """
    coordination = Coordination(sodium_silicate)
    speciation = coordination.get_bridging_speciation("O", "Si", cutoff=SI_O_CUTOFF)
    assert speciation[2] == pytest.approx(BRIDGING_OXYGEN_FRACTION, abs=0.01)
    assert speciation[1] == pytest.approx(1 - BRIDGING_OXYGEN_FRACTION, abs=0.01)


def test_mean_q_species_is_fixed_by_the_sodium_content(sodium_silicate):
    """<Qn> = 4 - 2*n(Na2O)/n(SiO2) = 3.143, the same count read from the silicon's side.

    Q3 must also be the most populous species at this composition, which is the shape of
    the distribution rather than just its mean.
    """
    coordination = Coordination(sodium_silicate)
    distribution = coordination.get_bridging_analysis("Si", "O", cutoff=SI_O_CUTOFF)
    mean_q = sum(n * fraction for n, fraction in distribution.items())
    assert mean_q == pytest.approx(MEAN_Q_SPECIES, abs=0.02)
    assert max(distribution, key=distribution.get) == 3


def test_bridging_analysis_and_speciation_count_the_same_bonds(sodium_silicate):
    """Bridges counted per tetrahedron and per oxygen are one quantity seen from two ends.

    Every bridging oxygen joins exactly two tetrahedra, so summing Qn over the silicons
    must give twice the number of bridging oxygens.
    """
    coordination = Coordination(sodium_silicate)
    per_silicon = coordination.get_bridging_analysis("Si", "O", cutoff=SI_O_CUTOFF, per_atom=True)
    per_oxygen = coordination.get_bridging_speciation("O", "Si", cutoff=SI_O_CUTOFF, per_atom=True)
    for silicons, oxygens in zip(per_silicon, per_oxygen):
        assert silicons.sum() == 2 * np.count_nonzero(oxygens == 2)


def test_intratetrahedral_angle_is_tetrahedral(sodium_silicate):
    """O-Si-O peaks at the tetrahedral 109.47 degrees, as a distribution and not a spike."""
    coordination = Coordination(sodium_silicate)
    angles, distribution = coordination.get_angle_distribution("Si", ["O", "O"], cutoff=SI_O_CUTOFF, range=(0.0, 180.0))
    assert angles[np.argmax(distribution)] == pytest.approx(109.47, abs=3.0)


def test_intertetrahedral_angle_is_the_silicate_one(sodium_silicate):
    """Si-O-Si is the flexible angle of the network and sits near 150 degrees.

    Centre and neighbour are the other way round from the test above, and the two must
    not come out the same: an implementation that ignored which species is the centre
    would report 109 degrees here.
    """
    coordination = Coordination(sodium_silicate)
    angles, distribution = coordination.get_angle_distribution(
        "O", ["Si", "Si"], cutoff=SI_O_CUTOFF, range=(0.0, 180.0)
    )
    assert angles[np.argmax(distribution)] == pytest.approx(150.0, abs=8.0)


def test_first_pdf_peak_is_the_si_o_bond(sodium_silicate):
    """g_SiO(r) peaks at the 1.60 A Si-O bond, with nothing at all below 1.3 A."""
    scattering = Scattering(sodium_silicate, rrange=8.0, nbin=800, disable_progress=True)
    pdf = scattering.get_partial_pdf(("Si", "O"))
    assert scattering.xval[np.argmax(pdf)] == pytest.approx(SI_O_BOND_LENGTH, abs=0.05)
    assert np.all(pdf[scattering.xval < 1.3] == 0.0)


def test_running_coordination_recovers_the_tetrahedron(sodium_silicate):
    """N_SiO(r) integrated past the first shell is 4, the same answer as counting bonds.

    The two routes to a coordination number -- integrating g(r) and counting neighbours
    inside a cutoff -- are independent code paths that must agree on a real structure.
    """
    scattering = Scattering(sodium_silicate, rrange=8.0, nbin=800, disable_progress=True)
    # 2.0 A is the minimum between the Si-O bond peak and everything beyond it.
    shell = np.searchsorted(scattering.xval, SI_O_CUTOFF)
    assert scattering.get_N_running(("Si", "O"))[shell] == pytest.approx(4.0, abs=0.1)


def test_silicate_rings_alternate_silicon_and_oxygen(sodium_silicate_frame):
    """Si and O alternate around any ring of a silicate network, so every ring is even.

    A topological consequence of the network being bipartite over Si-O bonds -- true of
    any silicate, at any composition -- rather than a count pinned to this structure.
    """
    rings = find_rings(
        sodium_silicate_frame,
        bonds=[("Si", "O")],
        limit=12,
        cutoff={("Si", "O"): SI_O_CUTOFF},
    )
    assert rings
    assert all(len(ring) % 2 == 0 for ring in rings)

    symbols = np.array(sodium_silicate_frame.get_chemical_symbols())
    for ring in rings:
        species = symbols[list(ring)]
        assert set(species) == {"Si", "O"}
        # Alternating means exactly half of each.
        assert np.count_nonzero(species == "Si") == len(ring) // 2


def test_the_melt_is_homogeneous(sodium_silicate_frame):
    """A melt has no phase separation to find; the checker must not invent one."""
    assert homogeneity_checker(sodium_silicate_frame, (3, 3, 3)) is True


def test_sodium_is_the_mobile_species(sodium_silicate_full):
    """Na diffuses through the silicate network while the network itself barely moves.

    Only the ordering is asserted. 100 ps from a single time origin is far too short to
    pin a diffusion coefficient -- it puts D(O) slightly negative -- which is the
    single-origin limitation recorded in docs/vitrum/known_issues.md.
    """
    times = np.arange(len(sodium_silicate_full), dtype=float)
    diffusion = Diffusion(sodium_silicate_full, list(times))
    coefficients = diffusion.get_diffusion_coef(skip_first=10)

    species = list(diffusion.species)
    d_na = coefficients[1 + species.index("Na")]
    d_si = coefficients[1 + species.index("Si")]
    assert d_na > 0
    assert d_na > 5 * abs(d_si)


def test_the_example_trajectory_needs_no_wrapped_override(sodium_silicate_full):
    """A LAMMPS dump writes wrapped coordinates that can land a hair outside the cell.

    The package's own example data does exactly that, so the default `wrapped=True` has
    to accept it rather than send users to a flag they do not need.
    """
    scaled = sodium_silicate_full[0].get_scaled_positions(wrap=False)
    assert scaled.min() < 0 or scaled.max() >= 1
    Diffusion(sodium_silicate_full, list(np.arange(len(sodium_silicate_full), dtype=float)))
