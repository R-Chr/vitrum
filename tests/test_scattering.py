"""Tests for vitrum.scattering.

The most important test here is test_neighborhood_matches_full_matrix: the two PDF
backends must agree, since they are two implementations of the same quantity.
"""

import numpy as np
import pytest

from vitrum.scattering import Scattering


@pytest.fixture(scope="module")
def scattering_pair(random_gas):
    """The same structure analysed with both PDF backends."""
    full = Scattering(random_gas, disable_progress=True, use_neighborhood=False)
    neigh = Scattering(random_gas, disable_progress=True, use_neighborhood=True)
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


@pytest.mark.parametrize("use_neighborhood", [False, True])
def test_partial_pdf_tends_to_unity(random_gas, use_neighborhood):
    """g(r) -> 1 at large r for a structure with no long-range correlation.

    This pins the normalisation: using N_a^2 rather than N_a*(N_a - 1) for like
    pairs makes the tail settle at (N_a - 1)/N_a instead of 1.
    """
    scattering = Scattering(
        random_gas, disable_progress=True, use_neighborhood=use_neighborhood
    )
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


def test_structure_factor_tends_to_unity(random_gas):
    """S(Q) -> 1 at high Q."""
    scattering = Scattering(random_gas, disable_progress=True)
    tail = scattering.get_structure_factor()[-100:].mean()
    assert tail == pytest.approx(1.0, abs=0.05)


def test_multi_frame_uses_per_frame_cell(random_gas):
    """Both backends must use each frame's own volume and symbols, not frame 0's.

    The two frames differ only by a 5% cell scaling, so a backend reading the cached
    self.volume or self.chemical_symbols instead of the frame's own shows up as a
    disagreement between the backends.
    """
    frame_a = random_gas.copy()
    frame_b = random_gas.copy()
    frame_b.set_cell(np.array(frame_b.get_cell()) * 1.05, scale_atoms=True)

    full = Scattering([frame_a, frame_b], disable_progress=True, use_neighborhood=False)
    neigh = Scattering([frame_a, frame_b], disable_progress=True, use_neighborhood=True)

    for pair in full.pairs:
        np.testing.assert_allclose(
            neigh.get_partial_pdf(pair), full.get_partial_pdf(pair),
            rtol=1e-6, atol=1e-9,
            err_msg=f"multi-frame PDF backends disagree for {pair}",
        )


def test_xray_rdf_raises_not_implemented(random_gas):
    """The unimplemented x-ray RDF must raise, not return an array of zeros."""
    scattering = Scattering(random_gas, disable_progress=True)
    with pytest.raises(NotImplementedError, match="not implemented"):
        scattering.get_total_rdf(type="xray")


def test_approx_xray_rdf_works(random_gas):
    """The approximate x-ray weighting is implemented and must still work."""
    scattering = Scattering(random_gas, disable_progress=True)
    rdf = scattering.get_total_rdf(type="approx_xray")
    assert rdf.shape == (scattering.nbin,)
    assert rdf[-100:].mean() == pytest.approx(1.0, abs=0.05)


def test_invalid_rdf_type_raises(random_gas):
    scattering = Scattering(random_gas, disable_progress=True)
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
    si_nn_distance = 5.431 * np.sqrt(3) / 4

    scattering = Scattering(silicon_diamond, disable_progress=True)
    pdf = scattering.get_partial_pdf(("Si", "Si"))
    first_shell = scattering.xval < 3.0
    peak_r = scattering.xval[first_shell][np.argmax(pdf[first_shell])]
    assert peak_r == pytest.approx(si_nn_distance, abs=0.05)

    # There must be nothing at all below the nearest-neighbour distance.
    assert np.all(pdf[scattering.xval < si_nn_distance - 0.1] == 0.0)


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


def test_average_density_is_trajectory_average(random_gas):
    """volume and aveden must average over frames, not snapshot frame 0.

    They feed get_N_running, the structure factors and the reduced PDFs, all of which
    would otherwise be normalised with the first frame's cell for an NPT trajectory.
    """
    frame_a = random_gas.copy()
    frame_b = random_gas.copy()
    frame_b.set_cell(np.array(frame_b.get_cell()) * 1.05, scale_atoms=True)

    scattering = Scattering([frame_a, frame_b], disable_progress=True)
    expected_volume = (frame_a.get_volume() + frame_b.get_volume()) / 2
    expected_density = (
        len(frame_a) / frame_a.get_volume() + len(frame_b) / frame_b.get_volume()
    ) / 2

    assert scattering.volume == pytest.approx(expected_volume)
    assert scattering.aveden == pytest.approx(expected_density)
    assert scattering.aveden != pytest.approx(len(frame_a) / frame_a.get_volume())


def test_varying_composition_is_rejected(random_gas):
    """Species counts are read from frame 0, so a changing composition must not be silent."""
    frame_a = random_gas.copy()
    frame_b = random_gas.copy()
    frame_b.symbols[0] = "O" if frame_b.symbols[0] == "Si" else "Si"

    with pytest.raises(ValueError, match="composition"):
        Scattering([frame_a, frame_b], disable_progress=True)


def test_glass_atoms_pdf_matches_scattering(silicon_diamond):
    """GlassAtoms.get_pdf and Scattering.get_partial_pdf are the same quantity.

    They are separate entry points to vitrum.geometry.pdf, so this pins them to a
    shared like-pair normalisation of N*(N-1); an N*N count in either would put them
    (N-1)/N apart.
    """
    from vitrum.glass_atoms import GlassAtoms

    atoms = GlassAtoms(silicon_diamond)
    _, direct = atoms.get_pdf(["Si", "Si"], rrange=8.0, nbin=400)
    scattering = Scattering([atoms], rrange=8.0, nbin=400, disable_progress=True)

    np.testing.assert_allclose(
        direct, scattering.get_partial_pdf(("Si", "Si")), rtol=1e-9, atol=1e-9
    )
