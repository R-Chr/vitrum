"""Tests for vitrum.scattering.

The most important test here is test_cell_list_matches_full_matrix: the two PDF backends
must agree, since they are two implementations of the same quantity.
"""

import numpy as np
import pytest
from ase.neighborlist import neighbor_list
from conftest import SI_NN_DISTANCE, SI_O_CUTOFF

from vitrum.geometry import cell_list_pair_counts, minimum_image_limit, perpendicular_widths
from vitrum.scattering import Scattering, gaussian_broadening


def with_backend(atoms, backend, **kwargs):
    """A `Scattering` whose partials come from `backend`.

    `Scattering` always uses the cell list. `_partial_pdfs_dense` is the O(N^2) route kept for
    this comparison alone -- it is private, and overwriting the partials is what these tests
    exist to do, not something a caller should be doing.
    """
    scattering = Scattering(atoms, **kwargs)
    scattering.partial_pdfs = (
        scattering._partial_pdfs_dense() if backend == "dense" else scattering.calculate_partial_pdfs_cell_list()
    )
    return scattering


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


# The O(N^2) reference is slow, so keep the range short here. Agreement with it is a
# per-frame, per-bin property, so one frame over the first few coordination shells shows any
# disagreement that a longer range would; `test_the_dense_reference_agrees_at_the_minimum
# _image_limit` covers the far end separately.
BACKEND_COMPARISON = dict(rrange=8.0, nbin=800, disable_progress=True)


@pytest.fixture(scope="module")
def scattering_pair(sodium_silicate_frame):
    """One frame of the melt analysed with both PDF backends."""
    full = with_backend(sodium_silicate_frame, "dense", **BACKEND_COMPARISON)
    neigh = with_backend(sodium_silicate_frame, "cell_list", **BACKEND_COMPARISON)
    return full, neigh


def test_backends_agree_on_a_triclinic_cell(sodium_silicate_triclinic):
    """Both backends must work on a general cell, and still agree there.

    They reach it by completely different routes -- the dense path through the numba
    minimum-image kernel, the cell list through its own fractional-shift stencil -- so
    agreement on a sheared cell is a real check on both and not a restatement of one code
    path.
    """
    full = with_backend(sodium_silicate_triclinic, "dense", **BACKEND_COMPARISON)
    neigh = with_backend(sodium_silicate_triclinic, "cell_list", **BACKEND_COMPARISON)
    for pair in full.pairs:
        from_full = full.get_partial_pdf(pair)
        assert np.any(from_full != 0.0), f"partial PDF for {pair} is all zeros"
        np.testing.assert_allclose(neigh.get_partial_pdf(pair), from_full, atol=1e-9)


def test_pdf_is_unchanged_by_a_unimodular_change_of_cell(sodium_silicate_frame):
    """The same lattice described by different cell vectors must give the same g(r).

    Replacing b with a + b is an integer basis change of determinant 1: it leaves the
    atoms and the infinite periodic structure completely untouched and only renames the
    cell, which happens to make it triclinic. Every minimum-image distance is therefore
    identical, so this pins the triclinic kernel to the orthorhombic one on a structure
    where the right answer is already known -- unlike shearing with `scale_atoms`, which
    is an affine map and genuinely does change the distances.
    """
    relabelled = sodium_silicate_frame.copy()
    basis = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    relabelled.set_cell(basis @ np.array(relabelled.get_cell()), scale_atoms=False)

    upright = Scattering(sodium_silicate_frame, **BACKEND_COMPARISON)
    retold = Scattering(relabelled, **BACKEND_COMPARISON)
    for pair in upright.pairs:
        np.testing.assert_allclose(retold.get_partial_pdf(pair), upright.get_partial_pdf(pair), atol=1e-9)


def test_default_rrange_is_half_the_cell_for_an_ordinary_cell(sodium_silicate_frame):
    """The 34.44 A example cell is well under the cap, so its default must not move."""
    scattering = Scattering(sodium_silicate_frame, disable_progress=True)
    assert scattering.rrange == pytest.approx(sodium_silicate_frame.get_cell().lengths().min() / 2)


def test_default_rrange_is_capped_for_a_large_cell(sodium_silicate_frame):
    """Past 40 A the default stops tracking the cell: the pair count grows as rrange^3."""
    big = sodium_silicate_frame.copy()
    big.set_cell(np.array(big.get_cell()) * 4, scale_atoms=True)
    assert Scattering(big, disable_progress=True).rrange == pytest.approx(20.0)


def test_explicit_rrange_is_not_capped(sodium_silicate_frame):
    """The 20 A cap is a default, not a limit; an explicit value below the minimum image
    limit is used exactly as given."""
    big = sodium_silicate_frame.copy()
    big.set_cell(np.array(big.get_cell()) * 4, scale_atoms=True)
    assert Scattering(big, rrange=30.0, nbin=100, disable_progress=True).rrange == pytest.approx(30.0)


def test_rrange_beyond_the_minimum_image_limit_is_rejected(sodium_silicate_frame):
    """Past half the shortest perpendicular width there is no honest g(r) to return.

    The distance matrix cannot represent a separation that long and empties towards
    g(r) -> 0; a pair list counts the periodic replicas instead and returns a
    plausible g(r) ~ 1 that is an artefact of the repetition. Neither is data.
    """
    limit = minimum_image_limit(sodium_silicate_frame.get_cell(), sodium_silicate_frame.pbc)
    with pytest.raises(ValueError, match=f"{limit:.2f}"):
        Scattering(sodium_silicate_frame, rrange=limit * 1.01, nbin=100, disable_progress=True)

    # Exactly at the limit is the default, so it has to stay legal.
    assert Scattering(sodium_silicate_frame, rrange=limit, nbin=100, disable_progress=True).rrange == pytest.approx(
        limit
    )


def test_the_cell_list_serves_the_whole_legal_rrange_range(sodium_silicate_frame):
    """There is no fallback, so the cell list has to cover every rrange `Scattering` accepts.

    The tightest case is the minimum image limit itself, where the 34.44 A cell holds only
    two cells per axis and the stencil wraps. Anything the constructor accepts must reach a
    result rather than a `ValueError` from inside the kernel.
    """
    limit = minimum_image_limit(sodium_silicate_frame.get_cell(), sodium_silicate_frame.pbc)
    for rrange in (limit / 6, limit / 3, limit):
        scattering = Scattering(sodium_silicate_frame, rrange=rrange, nbin=400, disable_progress=True)
        assert np.any(scattering.get_partial_pdf(("Si", "O")) != 0.0)


def test_the_dense_reference_agrees_at_the_minimum_image_limit(sodium_silicate_frame):
    """The two must agree where the cell list is most stressed, not just at short range.

    `scattering_pair` compares them at rrange 8, four cells per axis. This repeats it at the
    limit, two cells per axis, which is both the default for an ordinary cell and the case
    the wrapped stencil has to get right.
    """
    limit = minimum_image_limit(sodium_silicate_frame.get_cell(), sodium_silicate_frame.pbc)
    comparison = dict(rrange=limit, nbin=650, disable_progress=True)
    cell_list = Scattering(sodium_silicate_frame, **comparison)
    dense = with_backend(sodium_silicate_frame, "dense", **comparison)
    np.testing.assert_allclose(cell_list.get_total_rdf(), dense.get_total_rdf(), rtol=1e-6, atol=1e-9)


def test_non_periodic_input_is_rejected(silicon_small):
    """Both backends wrap unconditionally, so a free surface would be silently folded in."""
    cluster = silicon_small.copy()
    cluster.set_pbc(False)
    with pytest.raises(ValueError, match="periodic"):
        Scattering(cluster, disable_progress=True)


def test_default_rrange_uses_perpendicular_width_not_cell_length(sodium_silicate_triclinic):
    """A sheared cell is narrower than its axes are long, and the default must know it."""
    scattering = Scattering(sodium_silicate_triclinic, disable_progress=True)
    cell = sodium_silicate_triclinic.get_cell()
    assert scattering.rrange < cell.lengths().min() / 2
    assert scattering.rrange == pytest.approx(perpendicular_widths(cell).min() / 2)


def test_cell_list_matches_full_matrix_per_pair(scattering_pair):
    """Each partial PDF must be identical between the O(N^2) and cell-list paths.

    The all-zeros assertion is the point: the cell-list path keys its histograms by a
    sorted pair of species codes, so an unsorted lookup returns an empty row rather than
    an error, and one ordering of every cross pair comes back as exactly zero.
    """
    full, neigh = scattering_pair
    for pair in full.pairs:
        from_full = full.get_partial_pdf(pair)
        from_neigh = neigh.get_partial_pdf(pair)
        assert np.any(from_neigh != 0.0), f"partial PDF for {pair} is all zeros"
        np.testing.assert_allclose(
            from_neigh,
            from_full,
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"PDF backends disagree for pair {pair}",
        )


def test_cell_list_matches_full_matrix_total_rdf(scattering_pair):
    """The total RDF must not depend on which backend produced the partials."""
    full, neigh = scattering_pair
    np.testing.assert_allclose(neigh.get_total_rdf(), full.get_total_rdf(), rtol=1e-6, atol=1e-9)


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
    pairs makes the tail settle at (N_a - 1)/N_a instead of 1. The tail needs the full
    L/2 range;
    `test_cell_list_matches_full_matrix_per_pair` already pins the other one to this one
    bin by bin.
    """
    scattering = Scattering(sodium_silicate, disable_progress=True)
    for pair in scattering.pairs:
        tail = scattering.get_partial_pdf(pair)[-100:].mean()
        assert tail == pytest.approx(1.0, abs=0.03), f"{pair} tail g(r) = {tail}"


@pytest.mark.parametrize("backend", ["dense", "cell_list"])
def test_partial_pdf_matches_independent_reference(silicon_small, backend):
    """Pin the absolute normalisation against a direct, independent calculation.

    g_aa(r) = h(r) * V / (volbin(r) * N_a * (N_a - 1)), with h counting ordered pairs.
    The tail-tends-to-unity test cannot see the (N_a - 1)/N_a error at large N, so
    this reimplements the definition from ASE distances and compares exactly.
    """
    scattering = with_backend(silicon_small, backend, disable_progress=True)

    distances = silicon_small.get_all_distances(mic=True)
    n_atoms = len(silicon_small)
    off_diagonal = distances[~np.eye(n_atoms, dtype=bool)]
    counts, edges = np.histogram(off_diagonal, bins=scattering.nbin, range=(0.0, scattering.rrange))
    volbin = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    expected = (counts / volbin) / (n_atoms * (n_atoms - 1) / silicon_small.get_volume())

    # The final bin ends at exactly rrange = L/2, where the minimum image is ambiguous:
    # a separation of exactly half the cell length has two equally valid images. ASE's
    # neighbor_list counts such a pair once, the full distance matrix counts it twice.
    # That degeneracy is inherent to the MIC, not a property of either backend.
    np.testing.assert_allclose(scattering.get_partial_pdf(("Si", "Si"))[:-1], expected[:-1], rtol=1e-9, atol=1e-9)


def test_cell_list_counts_match_the_pair_list(sodium_silicate_frame):
    """The cell list must find exactly the pairs a pair list finds, in both directions.

    Normalisation would hide a factor of two here: halving every count and halving
    `n_pairs` gives the same g(r). This compares the raw counts instead, so the
    ordered-pair convention the two backends share is pinned on its own.
    """
    codes = np.zeros(len(sodium_silicate_frame), dtype=np.int64)
    counts = cell_list_pair_counts(sodium_silicate_frame, codes, 8.0, 800, 1)
    assert counts.sum() == len(neighbor_list("i", sodium_silicate_frame, 8.0))


@pytest.mark.parametrize(
    "fixture, rrange",
    [
        ("sodium_silicate_frame", 8.0),
        ("sodium_silicate_frame", 13.0),
        ("sodium_silicate_frame", 17.2),
        ("sodium_silicate_triclinic", 13.0),
        ("sodium_silicate_triclinic", 16.0),
    ],
)
def test_cell_list_counts_survive_a_stencil_that_wraps_onto_itself(request, fixture, rrange):
    """A stencil wider than the grid must not miscount. This is the default for most cells.

    The 34.44 A cell is cut into 8 cells at rrange = 8 A but only 4 at 16 A and at 17.2 A,
    the minimum image limit -- and a radius-2 stencil spans 5, so it wraps: the offsets +2
    and -2 reach the same cell with different shifts. Only +2 survives the halving, and the
    dropped image has to come back from the far end: i in cell 0 sees j in cell 2 at shift 0,
    and j sees i at shift +1, which is i seeing j at shift -1. The sheared cell is included
    because the shift bookkeeping is only non-trivial when the axes are not perpendicular.

    Compared per bin rather than by total, since a mirror-offset mistake can leave the total
    right and the bins wrong. Bin-exact agreement with a pair list is not guaranteed in
    principle -- a pair's two directions are different float expressions once a shift is
    involved, so one could round across a bin edge -- but that is measure-zero, and the total
    is preserved unconditionally.
    """
    atoms = request.getfixturevalue(fixture)
    nbin = 650
    codes = np.zeros(len(atoms), dtype=np.int64)
    counts = cell_list_pair_counts(atoms, codes, rrange, nbin, 1)
    expected, _ = np.histogram(neighbor_list("d", atoms, rrange), bins=nbin, range=(0.0, rrange))
    np.testing.assert_array_equal(counts[0], expected)


def test_cell_list_refuses_beyond_the_minimum_image_limit(sodium_silicate_frame):
    """Past half the shortest perpendicular width the counts would be periodic replicas.

    `Scattering` enforces this on `rrange` already; this is the guard for anyone calling
    the primitive directly, and it is the invariant the wrapped stencil above rests on.
    """
    codes = np.zeros(len(sodium_silicate_frame), dtype=np.int64)
    limit = minimum_image_limit(sodium_silicate_frame.get_cell(), sodium_silicate_frame.pbc)
    with pytest.raises(ValueError, match="minimum image limit"):
        cell_list_pair_counts(sodium_silicate_frame, codes, limit * 1.01, 650, 1)


def test_cell_list_refuses_a_non_periodic_cell(silicon_small):
    """A free axis has no images to wrap through, so the stencil has nothing to bin."""
    cluster = silicon_small.copy()
    cluster.set_pbc(False)
    codes = np.zeros(len(cluster), dtype=np.int64)
    with pytest.raises(ValueError, match="periodic"):
        cell_list_pair_counts(cluster, codes, 4.0, 400, 1)


def test_cell_list_refuses_codes_it_cannot_index(silicon_small):
    """A mis-sized or out-of-range `codes` must raise, not segfault.

    The kernel indexes `codes` by atom and the histogram by code, and numba bounds-checks
    neither -- either reads and writes off the end of a buffer, taking the interpreter down
    with it rather than raising.
    """
    with pytest.raises(ValueError, match="one entry per atom"):
        cell_list_pair_counts(silicon_small, np.zeros(3, dtype=np.int64), 4.0, 400, 1)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        cell_list_pair_counts(silicon_small, np.ones(len(silicon_small), dtype=np.int64), 4.0, 400, 1)


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

    full = with_backend([frame_a, frame_b], "dense", **BACKEND_COMPARISON)
    neigh = with_backend([frame_a, frame_b], "cell_list", **BACKEND_COMPARISON)

    for pair in full.pairs:
        np.testing.assert_allclose(
            neigh.get_partial_pdf(pair),
            full.get_partial_pdf(pair),
            rtol=1e-6,
            atol=1e-9,
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
    expected_density = (len(frame_a) / frame_a.get_volume() + len(frame_b) / frame_b.get_volume()) / 2

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

    np.testing.assert_allclose(direct, scattering.get_partial_pdf(("Si", "Si")), rtol=1e-9, atol=1e-9)
