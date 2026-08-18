"""Tests for vitrum.rings against known ring statistics.

Expected values are crystallographic, or come from brute-force enumeration of every
cycle in the bond graph, never from this code's own output. Where a test names a
R.I.N.G.S. routine, the value was checked against the Fortran sources of that routine.
"""

from collections import Counter

import numpy as np
import pytest
from ase import Atoms

from vitrum.rings import Ring, RingAnalysis, _fit_ellipse_axes, find_rings


def sizes(rings):
    """Ring-size histogram, for comparing against a known distribution."""
    return dict(Counter(len(ring) for ring in rings))


def ring_from(positions, cell=50.0):
    """A Ring holding the given coordinates, centred in a cell large enough not to matter."""
    positions = np.asarray(positions, dtype=float)
    return Ring(Atoms(f"Si{len(positions)}", positions=positions + cell / 2, cell=[cell] * 3, pbc=True))


def regular_polygon(n, radius=2.0):
    """Vertices of a regular n-gon of the given circumradius, in the xy plane."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)])


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


# --- the `cutoff` parameter: a bond graph from vitrum.bonds instead of covalent radii ------


def test_cutoff_none_matches_the_undecorated_default_call(silicon_small):
    """cutoff=None, the default, must be byte-identical to not passing it at all."""
    default = find_rings(silicon_small, bonds=None, limit=6)
    explicit_none = find_rings(silicon_small, bonds=None, limit=6, cutoff=None)
    assert sorted(sorted(r) for r in default) == sorted(sorted(r) for r in explicit_none)


def test_cutoff_dict_reproduces_the_radii_based_ring_graph(silicon_small):
    """A dict cutoff tuned to the same effective bond length must reproduce the same rings.

    Both build the same Si-Si bond graph here: the covalent-radii search finds the 2.3517 A
    nearest neighbours and nothing until the 3.84 A second shell, and a 3.0 A dict cutoff
    sits in the same gap. The ring *set*, not just the sizes, must match exactly.
    """
    from_radii = find_rings(silicon_small, bonds=None, limit=6)
    from_cutoff = find_rings(silicon_small, bonds=None, limit=6, cutoff={("Si", "Si"): 3.0})
    assert sorted(sorted(r) for r in from_radii) == sorted(sorted(r) for r in from_cutoff)


def test_cutoff_number_and_species_shorthand_agree_with_the_pair_dict(silicon_small):
    """The cutoff grammar's other spellings must resolve to the same graph as the pair dict."""
    by_pair = find_rings(silicon_small, bonds=None, limit=6, cutoff={("Si", "Si"): 3.0})
    by_number = find_rings(silicon_small, bonds=None, limit=6, cutoff=3.0)
    by_species = find_rings(silicon_small, bonds=None, limit=6, cutoff={"Si": 3.0})
    assert sizes(by_pair) == sizes(by_number) == sizes(by_species) == {6: 128}


def test_cutoff_auto_runs_end_to_end(silicon_small):
    """cutoff="Auto" must resolve on its own and reproduce diamond's known ring statistics."""
    rings = find_rings(silicon_small, bonds=None, limit=6, cutoff="Auto")
    assert sizes(rings) == {6: 128}


def test_cutoff_dict_restricts_to_the_named_bonds(silicon_small):
    """`bonds` still selects which species pairs enter the graph when `cutoff` is a dict."""
    rings = find_rings(silicon_small, bonds=[("Si", "Si")], limit=6, cutoff={("Si", "Si"): 3.0})
    assert sizes(rings) == {6: 128}


def test_ring_analysis_calculate_passes_cutoff_through(silicon_small):
    """RingAnalysis.calculate's cutoff must reach find_rings, not just be accepted."""
    analysis = RingAnalysis(silicon_small, included_atoms=["Si"])
    from_dict = analysis.calculate(max_size=6, cutoff={("Si", "Si"): 3.0})
    from_default = RingAnalysis(silicon_small, included_atoms=["Si"]).calculate(max_size=6)
    assert sizes([r.indexes for r in from_dict]) == sizes([r.indexes for r in from_default])


def test_unknown_criterion_raises(silicon_small):
    with pytest.raises(ValueError, match="Unknown ring criterion"):
        find_rings(silicon_small, bonds=None, criterion="nonsense")


@pytest.mark.parametrize("criterion", ["guttman", "king", "primitive"])
def test_limit_is_a_ring_size_for_every_criterion(silicon_small, criterion):
    """`limit` counts atoms in the ring, not Dijkstra hops.

    Every true ring in diamond has 6 atoms, so limit=5 must return nothing and limit=6
    must return them all — identically for each criterion. `limit` is passed to Dijkstra,
    which caps path *hops*, and the two differ by one for Guttman and two for
    King/primitive, so this previously admitted 6-rings at limit=4.
    """
    assert find_rings(silicon_small, bonds=None, limit=5, criterion=criterion) == []

    rings = find_rings(silicon_small, bonds=None, limit=6, criterion=criterion)
    assert len(rings) > 0
    assert {len(ring) for ring in rings} == {6}


@pytest.mark.parametrize("criterion", ["guttman", "king", "primitive"])
def test_silicon_ring_count_matches_crystallography(silicon_small, criterion):
    """Diamond has 12 six-rings per atom, so 64 atoms give 64 * 12 / 6 = 128 rings.

    Pinning the count, not just the sizes: the ring criteria are defined over *all*
    shortest paths, and taking a single Dijkstra predecessor chain per bond silently
    dropped the degenerate ones (98 for Guttman, 124 for King/primitive).
    """
    rings = find_rings(silicon_small, bonds=None, limit=6, criterion=criterion)
    assert len(rings) == 128
    assert {len(ring) for ring in rings} == {6}


def test_limit_below_three_raises(silicon_small):
    with pytest.raises(ValueError, match="maximum ring size"):
        find_rings(silicon_small, bonds=None, limit=2)


def test_primitive_rings_are_not_king_rings(simple_cubic):
    """Primitive rings cannot be found by filtering King's rings.

    Brute force over every non-wrapping cycle of the 27-atom simple-cubic bond graph,
    filtered by Franzblau's criterion, gives 81 four-rings and 108 six-rings. The
    six-rings run around a lattice cube; every atom on one has a two-hop shortcut
    between its two ring neighbours, so King never proposes them and generating King
    candidates first loses 57% of the answer.
    """
    rings = find_rings(simple_cubic, bonds=None, limit=6, repeat=(1, 1, 1), criterion="primitive")
    assert sizes(rings) == {4: 81, 6: 108}


def test_primitive_rings_in_the_cube_graph(cube_graph):
    """The cube graph Q3 is the smallest case where King's rings miss primitive ones.

    Six faces plus the four hexagons that wind around the cube, all primitive by brute
    force. King's criterion finds only the six faces.
    """
    rings = find_rings(cube_graph, bonds=None, limit=8, criterion="primitive")
    assert sizes(rings) == {4: 6, 6: 4}


def test_search_looks_past_rings_that_wrap_the_cell(silicon_cubic_cell):
    """A wrapping closure must not end the search for that bond.

    In the 8-atom cubic cell the minimum-image graph has 4-cycles that wind around the
    cell. They are not rings, but abandoning the bond there returns nothing at all;
    R.I.N.G.S. (`CHECK_LISTE`) treats them as dead branches and carries on, which
    recovers diamond's 16 six-rings (8 atoms * 12 rings per atom / 6).
    """
    rings = find_rings(silicon_cubic_cell, bonds=None, limit=8, repeat=(1, 1, 1), criterion="guttman")
    assert sizes(rings) == {6: 16}


def test_primitive_repeats_the_cell_by_default(simple_cubic):
    """A single cell cannot represent every primitive ring, so `repeat` defaults to 3x3x3.

    R.I.N.G.S. replicates unconditionally for this criterion (`dvtbox`); the answer has
    to come out the same as with an explicitly repeated cell.
    """
    default = find_rings(simple_cubic, bonds=None, limit=6, criterion="primitive")
    explicit = find_rings(simple_cubic, bonds=None, limit=6, repeat=(3, 3, 3), criterion="primitive")
    assert sizes(default) == sizes(explicit) == {4: 81, 6: 108}


def test_no_bonds_returns_no_rings():
    """An empty bond graph is not an error; it just has no rings.

    Building the sparse matrix from empty index arrays used to raise a ValueError from
    inside scipy.
    """
    from ase import Atoms

    isolated = Atoms("Ar4", positions=[(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)], cell=[30.0] * 3, pbc=True)
    assert find_rings(isolated, bonds=None, limit=8) == []


def test_ambiguous_periodic_bonds_warn():
    """A cell small enough to bond a pair through several images cannot be represented.

    The bond graph holds one edge per pair, so the extra images used to overwrite the
    offset map and make the periodicity check compare against an arbitrary offset.
    """
    from ase.build import bulk

    # The 2-atom primitive cell bonds atom 1 to atom 0 through three different images.
    primitive = bulk("Si", "diamond", a=5.431)

    with pytest.warns(UserWarning, match="more than one periodic image"):
        find_rings(primitive, criterion="guttman")


# --- ring topology: area, eccentricity, planeness (Supplemental note 1) ------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
def test_area_of_a_regular_polygon(n):
    """The center area of a regular n-gon is the exact polygon area.

    Summing the n center-to-edge triangles is the standard dissection of a convex polygon,
    so for a regular one it must reproduce (n/2) R^2 sin(2*pi/n).
    """
    radius = 2.0
    area = ring_from(regular_polygon(n, radius)).area()
    assert area == pytest.approx(0.5 * n * radius**2 * np.sin(2 * np.pi / n))


def test_area_is_independent_of_orientation_and_position():
    """Area is a shape property: rotating and translating the ring must not change it."""
    ring = regular_polygon(6)
    rotation = np.linalg.qr(np.random.default_rng(0).random((3, 3)))[0]
    assert ring_from(ring @ rotation.T + 3.7).area() == pytest.approx(ring_from(ring).area())


def test_eccentricity_of_a_regular_polygon_is_zero():
    """Every atom sits at the same distance from the center, so a = b and e = 0."""
    assert ring_from(regular_polygon(6)).eccentricity() == pytest.approx(0.0, abs=1e-6)


def test_eccentricity_uses_the_extreme_radii():
    """e = sqrt(1 - b^2/a^2) from the farthest and closest atom, computed by hand.

    Four atoms on the axes of an ellipse: the center is the origin, so a = 4 and b = 1
    directly, with no fitting involved.
    """
    ring = ring_from([[4, 0, 0], [0, 1, 0], [-4, 0, 0], [0, -1, 0]])
    assert ring.eccentricity() == pytest.approx(np.sqrt(1 - 1 / 16))


def test_planeness_of_a_flat_ring_is_zero():
    """A planar ring has no residual to its own best-fit plane, however it is oriented."""
    ring = regular_polygon(6)
    rotation = np.linalg.qr(np.random.default_rng(1).random((3, 3)))[0]
    assert ring_from(ring @ rotation.T).planeness() == pytest.approx(0.0, abs=1e-9)


def test_planeness_of_a_puckered_ring_is_the_mean_offset():
    """A square puckered by +-0.5 in z fits the z = 0 plane, leaving residuals of 0.5."""
    ring = ring_from([[0, 0, 0.5], [2, 0, -0.5], [2, 2, 0.5], [0, 2, -0.5]])
    assert ring.planeness() == pytest.approx(0.5)


def test_fitted_ellipse_recovers_known_axes():
    """The conic fit must return the semi-axes it was given, not just their ratio.

    Points sampled from an ellipse of known a and b, rotated and translated in the plane;
    an exact fit is expected because the points are noise-free.
    """
    rng = np.random.default_rng(3)
    for a, b in [(3.0, 1.0), (1.0, 1.0), (5.0, 4.5), (2.7, 0.3)]:
        angles = np.sort(rng.random(9)) * 2 * np.pi
        points = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
        phi = rng.random() * np.pi
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        points = points @ rotation.T + rng.random(2) * 10
        assert _fit_ellipse_axes(points[:, 0], points[:, 1]) == pytest.approx((a, b))


def test_ellipse_eccentricity_of_a_tilted_ellipse():
    """Projecting onto the best-fit plane must undo an arbitrary 3D tilt of a planar ring."""
    a, b = 3.0, 1.0
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    ring = np.column_stack([a * np.cos(angles), b * np.sin(angles), np.zeros(9)])
    rotation = np.linalg.qr(np.random.default_rng(0).random((3, 3)))[0]
    assert ring_from(ring @ rotation.T).ellipse_eccentricity() == pytest.approx(np.sqrt(1 - b**2 / a**2))


def test_ellipse_eccentricity_of_a_regular_polygon_is_zero():
    """A regular polygon's best-fit ellipse is a circle."""
    assert ring_from(regular_polygon(6)).ellipse_eccentricity() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n", [3, 4])
def test_ellipse_eccentricity_is_undefined_below_five_atoms(n):
    """Five points fix a conic; fewer leave the ellipse undetermined, so the result is NaN.

    `eccentricity` has no such restriction and must still give an answer for these rings.
    """
    ring = ring_from(regular_polygon(n))
    assert np.isnan(ring.ellipse_eccentricity())
    assert not np.isnan(ring.eccentricity())


def test_ellipse_eccentricity_is_nan_for_a_degenerate_ring():
    """Collinear atoms determine no ellipse; the fit must decline rather than raise."""
    assert np.isnan(ring_from([[i, 0.0, 0.0] for i in range(6)]).ellipse_eccentricity())


def test_topology_is_unaffected_by_the_periodic_boundary():
    """A ring straddling a cell face must measure the same as the same ring inside it.

    The metrics all work from `_unwrapped_positions`, so this is what stops a ring being
    reported as huge and misshapen purely because of where it sits in the cell.
    """
    ring = regular_polygon(6)
    inside = Ring(Atoms("Si6", positions=ring + 5, cell=[10] * 3, pbc=True))
    straddling = Ring(Atoms("Si6", positions=(ring + 0.2) % 10, cell=[10] * 3, pbc=True))
    for metric in ("area", "eccentricity", "planeness", "ellipse_eccentricity", "perimeter"):
        assert getattr(straddling, metric)() == pytest.approx(getattr(inside, metric)(), abs=1e-6)


@pytest.mark.parametrize("n,radius", [(12, 4.5), (8, 4.2)])
def test_topology_of_a_ring_wider_than_half_the_cell(n, radius):
    """A ring wider than L/2 must still measure as the polygon it is.

    The ring here sits entirely inside the cell and touches no face, but spans more than
    half of it, so each atom's minimum image taken from the first atom lands on the wrong
    side. Unwrapping bond by bond is what keeps the metrics exact.
    """
    box = 12.0
    ring = ring_from(regular_polygon(n, radius), cell=box)
    assert ring.perimeter() == pytest.approx(n * 2 * radius * np.sin(np.pi / n))
    assert ring.area() == pytest.approx(0.5 * n * radius**2 * np.sin(2 * np.pi / n))
    assert ring.radius_of_gyration() == pytest.approx(radius)
    assert ring.roundness() == pytest.approx(1.0)
    assert ring.planeness() == pytest.approx(0.0, abs=1e-9)


def test_wide_ring_topology_is_unaffected_by_the_periodic_boundary():
    """The L/2-spanning ring must also measure the same when moved across a cell face."""
    box = 12.0
    polygon = regular_polygon(12, 4.5)
    inside = Ring(Atoms("Si12", positions=polygon + box / 2, cell=[box] * 3, pbc=True))
    straddling = Ring(Atoms("Si12", positions=(polygon + 0.3) % box, cell=[box] * 3, pbc=True))
    for metric in ("area", "perimeter", "roundness", "planeness", "radius_of_gyration"):
        assert getattr(straddling, metric)() == pytest.approx(getattr(inside, metric)(), abs=1e-6)


@pytest.mark.parametrize("criterion", ["guttman", "king", "primitive"])
def test_rings_folding_onto_their_own_image_are_rejected(criterion):
    """A ring larger than the primary cell cannot be expressed in its indices.

    One atom per cell means every ring found in the repeated cell folds back onto index 0
    repeatedly, which is not a ring of the primary cell and must be dropped rather than
    reported as an N-ring through a single atom.
    """
    cell = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[2.4] * 3, pbc=True)
    with pytest.warns(UserWarning, match="more than one periodic image"):
        rings = find_rings(cell, repeat=(3, 3, 3), criterion=criterion, limit=6)
    assert all(len(set(ring)) == len(ring) for ring in rings)


def test_topology_metrics_on_rings_from_a_real_structure(silicon_small):
    """Every metric must return a finite, sensible number for real six-membered rings.

    Guards against a metric that only works on the synthetic planar rings above: the
    six-rings of diamond silicon are puckered, so planeness must be non-zero while the
    area stays bounded by that of a regular hexagon with the same perimeter.
    """
    rings = [Ring(silicon_small[list(r)], list(r)) for r in find_rings(silicon_small, bonds=None, limit=6)]
    assert rings

    for ring in rings:
        assert ring.size() == 6
        assert ring.area() > 0
        assert 0.0 <= ring.eccentricity() <= 1.0
        assert ring.planeness() > 0.0  # the chair conformation is not flat
        assert 0.0 <= ring.ellipse_eccentricity() <= 1.0
        side = ring.perimeter() / 6
        assert ring.area() < 1.5 * np.sqrt(3) * side**2  # regular hexagon of the same perimeter
