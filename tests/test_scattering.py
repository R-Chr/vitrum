"""Tests for vitrum.scattering.

The most important test here is test_neighborhood_matches_full_matrix: the two PDF
backends must agree, since they are two implementations of the same quantity.
"""

import numpy as np
import pytest
from conftest import SI_NN_DISTANCE, SI_O_CUTOFF

from vitrum.scattering import Scattering, gaussian_broadening


def test_broadening_conserves_the_area_of_a_shell():
    """Q_max broadening blurs a coordination shell; it must not create or destroy atoms.

    The kernel is the truncation broadening of the odd function r*g(r), so applying it to
    g(r) directly would leave a Q_max-dependent error in the coordination integral.
    """
    r = np.linspace(0.01, 10, 2000)
    shell = np.exp(-((r - 1.6) ** 2) / (2 * 0.03**2))
    reference = np.trapezoid(r**2 * shell, r)
    for q_max in (30.0, 15.0, 8.0, 5.0):
        broadened = np.trapezoid(r**2 * gaussian_broadening(shell, r, q_max), r)
        assert broadened == pytest.approx(reference, rel=1e-3)


def test_broadening_leaves_the_asymptote_alone():
    """g(r) -> 1 away from the origin, and broadening must not shift that baseline."""
    r = np.linspace(0.01, 10, 500)
    broadened = gaussian_broadening(np.ones_like(r), r, 20.0)
    np.testing.assert_allclose(np.interp([2.0, 5.0, 9.0], r, broadened), 1.0, atol=1e-6)


def test_untabulated_neutron_scattering_length_names_the_element(silicon_small):
    """A missing scattering length must say which element, not fail inside numpy."""
    thorium = silicon_small.copy()
    thorium.symbols = ["Th"] * len(thorium)
    with pytest.raises(ValueError, match="Th"):
        Scattering(thorium, disable_progress=True)


# The neighbourhood backend costs ~30x the dense one on 3000 atoms, and grows as rrange^3.
# Backend agreement is a per-frame, per-bin property, so one frame over the first few
# coordination shells shows any disagreement that a longer range would.
BACKEND_COMPARISON = dict(rrange=8.0, nbin=800, disable_progress=True)


@pytest.fixture(scope="module")
def scattering_pair(sodium_silicate_frame):
    """One frame of the melt analysed with both PDF backends."""
    full = Scattering(sodium_silicate_frame, use_neighborhood=False, **BACKEND_COMPARISON)
    neigh = Scattering(sodium_silicate_frame, use_neighborhood=True, **BACKEND_COMPARISON)
    return full, neigh


def test_neighborhood_matches_full_matrix_per_pair(scattering_pair):
    """Each partial PDF must be identical between the O(N^2) and neighbour-list paths.

    The all-zeros assertion is the point: the neighbour path keys its distances by a
    sorted element tuple, so an unsorted lookup returns an empty list rather than an
    error, and one ordering of every cross pair comes back as exactly zero.
    """
    full, neigh = scattering_pair
    for pair in full.pairs:
        from_full = full.get_partial_pdf(pair)
        from_neigh = neigh.get_partial_pdf(pair)
        assert np.any(from_neigh != 0.0), f"partial PDF for {pair} is all zeros"
        np.testing.assert_allclose(
            from_neigh, from_full, rtol=1e-6, atol=1e-9,
            err_msg=f"PDF backends disagree for pair {pair}",
        )


def test_neighborhood_matches_full_matrix_total_rdf(scattering_pair):
    """The total RDF must not depend on which backend produced the partials."""
    full, neigh = scattering_pair
    np.testing.assert_allclose(
        neigh.get_total_rdf(), full.get_total_rdf(), rtol=1e-6, atol=1e-9
    )


def test_cross_pairs_are_symmetric(scattering_pair):
    """g_ij(r) == g_ji(r) for both backends."""
    for scattering in scattering_pair:
        np.testing.assert_allclose(
            scattering.get_partial_pdf(("Si", "O")),
            scattering.get_partial_pdf(("O", "Si")),
            rtol=1e-9,
        )


def test_partial_pdf_tends_to_unity(sodium_silicate):
    """g(r) -> 1 at large r, which is what a glass does and a crystal never does.

    This pins the normalisation: using N_a^2 rather than N_a*(N_a - 1) for like
    pairs makes the tail settle at (N_a - 1)/N_a instead of 1. Only the dense backend
    is run: the tail needs the full L/2 range, which is expensive on the neighbourhood
    path, and `test_neighborhood_matches_full_matrix_per_pair` already pins that one to
    this one bin by bin.
    """
    scattering = Scattering(sodium_silicate, disable_progress=True)
    for pair in scattering.pairs:
        tail = scattering.get_partial_pdf(pair)[-100:].mean()
        assert tail == pytest.approx(1.0, abs=0.03), f"{pair} tail g(r) = {tail}"


@pytest.mark.parametrize("use_neighborhood", [False, True])
def test_partial_pdf_matches_independent_reference(silicon_small, use_neighborhood):
    """Pin the absolute normalisation against a direct, independent calculation.

    g_aa(r) = h(r) * V / (volbin(r) * N_a * (N_a - 1)), with h counting ordered pairs.
    The tail-tends-to-unity test cannot see the (N_a - 1)/N_a error at large N, so
    this reimplements the definition from ASE distances and compares exactly.
    """
    scattering = Scattering(
        silicon_small, disable_progress=True, use_neighborhood=use_neighborhood
    )

    distances = silicon_small.get_all_distances(mic=True)
    n_atoms = len(silicon_small)
    off_diagonal = distances[~np.eye(n_atoms, dtype=bool)]
    counts, edges = np.histogram(
        off_diagonal, bins=scattering.nbin, range=(0.0, scattering.rrange)
    )
    volbin = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    expected = (counts / volbin) / (n_atoms * (n_atoms - 1) / silicon_small.get_volume())

    # The final bin ends at exactly rrange = L/2, where the minimum image is ambiguous:
    # a separation of exactly half the cell length has two equally valid images. ASE's
    # neighbor_list counts such a pair once, the full distance matrix counts it twice.
    # That degeneracy is inherent to the MIC, not a property of either backend.
    np.testing.assert_allclose(
        scattering.get_partial_pdf(("Si", "Si"))[:-1], expected[:-1], rtol=1e-9, atol=1e-9
    )


def test_structure_factor_tends_to_unity(sodium_silicate):
    """S(Q) -> 1 at high Q."""
    scattering = Scattering(sodium_silicate, disable_progress=True)
    tail = scattering.get_structure_factor()[-100:].mean()
    assert tail == pytest.approx(1.0, abs=0.05)


def test_multi_frame_uses_per_frame_cell(sodium_silicate):
    """Both backends must use each frame's own volume and symbols, not frame 0's.

    Two genuinely different configurations from the melt, with the second one's cell
    scaled by 5% on top. A backend reading the cached self.volume or
    self.chemical_symbols instead of the frame's own shows up as a disagreement between
    the backends.
    """
    frame_a = sodium_silicate[0].copy()
    frame_b = sodium_silicate[-1].copy()
    frame_b.set_cell(np.array(frame_b.get_cell()) * 1.05, scale_atoms=True)

    full = Scattering([frame_a, frame_b], use_neighborhood=False, **BACKEND_COMPARISON)
    neigh = Scattering([frame_a, frame_b], use_neighborhood=True, **BACKEND_COMPARISON)

    for pair in full.pairs:
        np.testing.assert_allclose(
            neigh.get_partial_pdf(pair), full.get_partial_pdf(pair),
            rtol=1e-6, atol=1e-9,
            err_msg=f"multi-frame PDF backends disagree for {pair}",
        )


def test_xray_rdf_matches_the_q_space_transform(sodium_silicate_frame):
    """G^X(r) must be the Fourier partner of the x-ray S(Q) the class already computes.

    Keen eq 58, G^X(r) = (1 / (2 pi^2 rho_0 r)) Integral[ Q (S(Q) - 1) sin(Qr) dQ ], is
    written out here and shares no code with the r-space path, so agreement pins the kernel
    normalisation and the odd extension of r*(g-1). qmin is 0 to cover exactly the [0, qmax]
    range the truncated eq 61 kernel is built from.
    """
    scattering = Scattering(sodium_silicate_frame, qmin=0.0, qmax=25.0, disable_progress=True)
    q, r = scattering.qval, scattering.xval

    f_q = q * (scattering.get_structure_factor(type="xray") - 1.0)
    from_q_space = np.trapezoid(f_q * np.sin(np.outer(r, q)), q) / (2 * np.pi**2 * scattering.aveden * r)

    from_r_space = scattering.get_total_rdf(type="xray") - 1.0
    # Not masking r below the first bond: a misnormalised kernel shows up there first.
    np.testing.assert_allclose(from_r_space, from_q_space, atol=0.02)


def test_lorch_suppresses_the_truncation_ripple(sodium_silicate_frame):
    """Below the shortest bond the true G'(r) is 0, so anything there is truncation ripple."""
    scattering = Scattering(sodium_silicate_frame, qmax=25.0, disable_progress=True)
    below_first_bond = scattering.xval < 1.4

    plain = scattering.get_total_rdf(type="xray")
    lorched = scattering.get_total_rdf(type="xray", lorch=True)

    ripple = np.sqrt(np.mean(lorched[below_first_bond] ** 2))
    assert ripple < 0.15
    assert ripple < np.sqrt(np.mean(plain[below_first_bond] ** 2)) / 10

    # SI_O_CUTOFF is past the Si-O bond and short of any second shell.
    peak_r = scattering.xval[np.argmax(lorched[scattering.xval < SI_O_CUTOFF])]
    assert peak_r == pytest.approx(1.60, abs=0.05)

    # Broadening genuine features is the price, so the peak must survive rather than vanish.
    assert lorched.max() > 1.5
    assert lorched[-100:].mean() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("type", ["neutron", "approx_xray"])
def test_lorch_rejected_where_there_is_no_transform(sodium_silicate_frame, type):
    scattering = Scattering(sodium_silicate_frame, disable_progress=True)
    with pytest.raises(ValueError, match="lorch"):
        scattering.get_total_rdf(type=type, lorch=True)


def test_xray_rdf_is_not_the_atomic_number_approximation(sodium_silicate_frame):
    """approx_xray weights by Z_i = f_i(0); agreement would mean f_i was evaluated at one Q."""
    scattering = Scattering(sodium_silicate_frame, disable_progress=True)
    xray = scattering.get_total_rdf(type="xray")
    assert xray.shape == (scattering.nbin,)
    assert np.all(np.isfinite(xray))
    # g_ij(r) -> 1 for every pair, so every weighting scheme shares the same asymptote.
    assert xray[-100:].mean() == pytest.approx(1.0, abs=0.05)
    assert not np.allclose(xray, scattering.get_total_rdf(type="approx_xray"), atol=0.05)


def test_approx_xray_rdf_works(sodium_silicate_frame):
    """The approximate x-ray weighting is implemented and must still work."""
    scattering = Scattering(sodium_silicate_frame, disable_progress=True)
    rdf = scattering.get_total_rdf(type="approx_xray")
    assert rdf.shape == (scattering.nbin,)
    assert rdf[-100:].mean() == pytest.approx(1.0, abs=0.05)


def test_invalid_rdf_type_raises(sodium_silicate_frame):
    scattering = Scattering(sodium_silicate_frame, disable_progress=True)
    with pytest.raises(ValueError):
        scattering.get_total_rdf(type="not-a-type")


def test_triclinic_cell_rejected(triclinic_atoms):
    with pytest.raises(NotImplementedError, match="orthorhombic"):
        Scattering(triclinic_atoms, disable_progress=True)


def test_silicon_first_peak_at_nn_distance(silicon_diamond):
    """The first PDF peak in crystalline Si must sit at the known 2.3517 A.

    Only the first shell is considered: the 12-neighbour second shell at 3.840 A is
    actually the taller peak in g(r), so a global argmax would find that instead.
    """
    scattering = Scattering(silicon_diamond, disable_progress=True)
    pdf = scattering.get_partial_pdf(("Si", "Si"))
    first_shell = scattering.xval < 3.0
    peak_r = scattering.xval[first_shell][np.argmax(pdf[first_shell])]
    assert peak_r == pytest.approx(SI_NN_DISTANCE, abs=0.05)

    # There must be nothing at all below the nearest-neighbour distance.
    assert np.all(pdf[scattering.xval < SI_NN_DISTANCE - 0.1] == 0.0)


def test_running_coordination_uses_neighbour_density(fluorite_caf2):
    """N_ij(r) = 4*pi*rho_j*Integral(g_ij r^2 dr) is normalised by the neighbour species.

    Fluorite CaF2 has an asymmetric ground truth -- Ca is 8-coordinated by F, F is
    4-coordinated by Ca -- so swapping the centre and neighbour species is unmissable
    here, and invisible in any 1:1 stoichiometry.
    """
    scattering = Scattering(fluorite_caf2, rrange=8.0, nbin=800, disable_progress=True)
    # 3.0 A sits in the gap between the first (2.36 A) and second (3.86 A) Ca-F shells.
    shell = np.searchsorted(scattering.xval, 3.0)

    assert scattering.get_N_running(("Ca", "F"))[shell] == pytest.approx(8.0, abs=0.05)
    assert scattering.get_N_running(("F", "Ca"))[shell] == pytest.approx(4.0, abs=0.05)
    # F-F: 6 neighbours at a/2 = 2.73 A. Ca-Ca: nothing until a/sqrt(2) = 3.85 A.
    assert scattering.get_N_running(("F", "F"))[shell] == pytest.approx(6.0, abs=0.05)
    assert scattering.get_N_running(("Ca", "Ca"))[shell] == pytest.approx(0.0, abs=1e-9)


def test_average_density_is_trajectory_average(sodium_silicate_frame):
    """volume and aveden must average over frames, not snapshot frame 0.

    They feed get_N_running, the structure factors and the reduced PDFs, all of which
    would otherwise be normalised with the first frame's cell for an NPT trajectory.
    """
    frame_a = sodium_silicate_frame.copy()
    frame_b = sodium_silicate_frame.copy()
    frame_b.set_cell(np.array(frame_b.get_cell()) * 1.05, scale_atoms=True)

    scattering = Scattering([frame_a, frame_b], disable_progress=True)
    expected_volume = (frame_a.get_volume() + frame_b.get_volume()) / 2
    expected_density = (
        len(frame_a) / frame_a.get_volume() + len(frame_b) / frame_b.get_volume()
    ) / 2

    assert scattering.volume == pytest.approx(expected_volume)
    assert scattering.aveden == pytest.approx(expected_density)
    assert scattering.aveden != pytest.approx(len(frame_a) / frame_a.get_volume())


def test_varying_composition_is_rejected(sodium_silicate_frame):
    """Species counts are read from frame 0, so a changing composition must not be silent."""
    frame_a = sodium_silicate_frame.copy()
    frame_b = sodium_silicate_frame.copy()
    frame_b.symbols[0] = "O" if frame_b.symbols[0] == "Si" else "Si"

    with pytest.raises(ValueError, match="composition"):
        Scattering([frame_a, frame_b], disable_progress=True)


def test_geometry_partial_pdf_matches_scattering(silicon_diamond):
    """geometry.partial_pdf and Scattering.get_partial_pdf are the same quantity.

    Scattering's dense backend is a frame-averaging loop over partial_pdf, so this pins
    the single-frame case to it, and with it the like-pair normalisation of N*(N-1); an
    N*N count in either would put them (N-1)/N apart.
    """
    from vitrum.geometry import distance_matrix, partial_pdf

    _, direct = partial_pdf(
        distance_matrix(silicon_diamond),
        silicon_diamond.get_chemical_symbols(),
        silicon_diamond.get_volume(),
        ("Si", "Si"),
        rrange=8.0,
        nbin=400,
    )
    scattering = Scattering([silicon_diamond], rrange=8.0, nbin=400, disable_progress=True)

    np.testing.assert_allclose(
        direct, scattering.get_partial_pdf(("Si", "Si")), rtol=1e-9, atol=1e-9
    )
