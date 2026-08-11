from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce
from numba import get_num_threads, njit, prange
from numpy.typing import ArrayLike
from scipy.signal import argrelextrema

# Cells per `rrange` along each axis in the cell list; see `cell_list_pair_counts`. Affects
# speed only, never counts. Measured fastest at 2 over the rrange range `Scattering` uses.
# ponytail: one constant for every cell and cutoff. With the cell sort it costs ~2 ms/frame
# at 16000 atoms and rrange = 4 A, where cells are near-empty. Pick it from the occupancy if
# a short cutoff ever becomes a real workload.
_GRID_REFINE = 2


def perpendicular_widths(cell: ArrayLike) -> np.ndarray:
    """
    Distance between each pair of opposite cell faces, (w_a, w_b, w_c).

    Half the smallest of these is the largest radius the minimum image convention holds
    for. The cell diagonal only reports it correctly for an orthorhombic cell: shearing an
    axis leaves its length unchanged but brings the faces it spans closer together, so the
    diagonal over-estimates how far a triclinic cell can be sampled.

    Args:
        cell (array_like): A 3x3 cell matrix, e.g. from ``atoms.get_cell()``.

    Returns:
        np.ndarray: The perpendicular width along each axis, shape (3,).

    Raises:
        np.linalg.LinAlgError: If the cell is degenerate (zero volume).
    """
    # w_a = V / |b x c|, and the columns of the inverse are (b x c) / V and its cyclic
    # permutations, so each column norm is one over a width.
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, dtype=float)), axis=0)


def _reduced_cell(cell: ArrayLike, pbc: ArrayLike) -> np.ndarray:
    """Minkowski-reduced cell, as a plain array: `minkowski_reduce` hands back an
    `ase.cell.Cell`, which numba cannot type."""
    reduced, _ = minkowski_reduce(np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool))
    return np.ascontiguousarray(reduced, dtype=float)


def minimum_image_limit(cell: ArrayLike, pbc: ArrayLike) -> float:
    """
    The largest radius at which the minimum image convention still holds for a cell.

    Every pair separation shorter than this has exactly one periodic image; past it a
    neighbour and its replica are both in range and any g(r) counts one of them twice.

    Both the cell as given and its Minkowski reduction bound this, and neither is reliably
    the tighter of the two: reduction shortens the basis vectors, but shortening one can
    narrow the cell across a face -- measured at up to 10% narrower over random skewed cells.
    The smaller of the two is therefore the bound that holds for the caller's own cell *and*
    for the reduced basis a cell list is gridded on, which keeps the two from disagreeing
    about which radii are legal.

    Args:
        cell (array_like): A 3x3 cell matrix, e.g. from ``atoms.get_cell()``.
        pbc (array_like): Per-axis periodicity, e.g. ``atoms.pbc``.

    Returns:
        float: Half the shortest perpendicular width, over both bases.
    """
    return float(min(perpendicular_widths(cell).min(), perpendicular_widths(_reduced_cell(cell, pbc)).min())) / 2


def require_orthorhombic(cell: ArrayLike, caller: str = "") -> np.ndarray:
    """
    Return the cell diagonal, raising if the cell is not orthorhombic.

    Much of vitrum assumes an orthorhombic cell, applying the minimum image convention
    with the cell diagonal only. Passing a triclinic cell to those routines yields
    plausible-looking but wrong numbers, so it is rejected here instead.

    Args:
        cell (array_like): A 3x3 cell matrix, e.g. from ``atoms.get_cell()``.
        caller (str, optional): Name of the calling function, used in the error message.

    Returns:
        np.ndarray: The cell dimensions (lx, ly, lz).

    Raises:
        NotImplementedError: If the cell has non-zero off-diagonal components.
    """
    c = np.asarray(cell, dtype=float)
    if not np.allclose(c - np.diag(np.diagonal(c)), 0.0, atol=1e-8):
        raise NotImplementedError(
            f"{caller or 'This function'} assumes an orthorhombic cell "
            "(off-diagonal cell components must be zero), but got a triclinic cell:\n"
            f"{c}"
        )
    return np.diagonal(c).copy()


def find_min_after_peak(padf: ArrayLike, context: str = "") -> int:
    """
    Find the index of the first local minimum after the first peak in a function.
    Useful for determining cutoffs from PDFs.

    The first peak is located as the first local maximum, so the empty bins below the
    closest approach are skipped however they are distributed.

    Args:
        padf (np.ndarray): The probability density function or similar array.
        context (str, optional): Description of what is being analysed (e.g. the atom
            pair), used to make the error message actionable.

    Returns:
        int: The index of the minimum.

    Raises:
        ValueError: If the function has no local minimum after a first peak
    """
    padf = np.asarray(padf)
    peaks = argrelextrema(padf, np.greater, order=4)[0]
    mins = argrelextrema(padf, np.less_equal, order=4)[0]
    after_peak = mins[mins > peaks[0]] if peaks.size else np.array([], dtype=int)
    if after_peak.size == 0:
        raise ValueError(
            f"Could not determine an automatic cutoff{f' for {context}' if context else ''}: "
            "the distribution has no local minimum after a first peak. Pass an explicit `cutoff` instead."
        )
    return int(after_peak[0])


def peak_metrics(x: np.ndarray, y: np.ndarray, window: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """
    Position, width and height of the first peak of a tabulated function.

    The first peak is the first local maximum, found the same way as in
    `find_min_after_peak`. The width is the full width at half maximum, taken between the
    two half-height crossings either side of the peak by linear interpolation, so it is not
    limited to the bin spacing. Height is measured from zero, not from a fitted baseline.

    This is the one primitive behind several reported quantities: on a partial g(r) it gives
    the bond length and the static disorder of that bond, and on S(Q) it gives the position,
    width and intensity of the first sharp diffraction peak.

    Args:
        x (np.ndarray): The abscissa, increasing.
        y (np.ndarray): The function values, same length as `x`.
        window (Optional[Tuple[float, float]], optional): Restrict the search to this range
            of `x`, for picking out a peak that is not the first one overall. Defaults to
            None, meaning the whole range. Simulated S(Q) usually needs one: the noise
            below the first sharp diffraction peak carries local maxima of its own, and
            they come first.

    Returns:
        Tuple[float, float, float]: The peak position in units of `x`, its full width at
            half maximum, and its height. The width is NaN where the data do not fall to
            half maximum on both sides, which is the sign that the peak runs off the end of
            the range or the window is too tight.

    Raises:
        ValueError: If `x` and `y` differ in length, or the searched range holds no local
            maximum.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    if window is not None:
        inside = (x >= window[0]) & (x <= window[1])
        x, y = x[inside], y[inside]

    peaks = argrelextrema(y, np.greater, order=4)[0]
    if peaks.size == 0:
        raise ValueError(
            "No local maximum found"
            f"{f' within {window}' if window is not None else ''}: the function is "
            "monotonic or too noisy to peak-find. Widen the window or smooth the input."
        )
    top = int(peaks[0])
    height = float(y[top])

    half = height / 2
    left = _crossing(x, y, top, half, step=-1)
    right = _crossing(x, y, top, half, step=1)
    fwhm = right - left if (left is not None and right is not None) else float("nan")
    return float(x[top]), float(fwhm), height


def _crossing(x: np.ndarray, y: np.ndarray, start: int, level: float, step: int) -> float | None:
    """Where `y` first falls to `level` walking from `start`, interpolated; None if never."""
    for i in range(start, -1 if step < 0 else len(y) - 1, step):
        nxt = i + step
        if not 0 <= nxt < len(y):
            return None
        if y[nxt] <= level:
            span = y[nxt] - y[i]
            fraction = 0.0 if span == 0 else (level - y[i]) / span
            return float(x[i] + fraction * (x[nxt] - x[i]))
    return None


def radial_bins(rrange: float = 10, nbin: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin centres and shell volumes for a radial histogram.

    Parameters:
        rrange (float, optional): The range of the histogram. Defaults to 10.
        nbin (int, optional): The number of bins. Defaults to 100.

    Returns:
        xval (np.ndarray): Bin centres, shape (nbin,).
        volbin (np.ndarray): Volume of each spherical shell, shape (nbin,).
    """
    edges = np.linspace(0, rrange, nbin + 1)
    xval = (edges[1:] + edges[:-1]) / 2
    volbin = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return xval, volbin


def pdf(
    dist_list: ArrayLike | None,
    volume: float,
    rrange: float = 10,
    nbin: int = 100,
    n_pairs: int | None = None,
    exclude_self: bool = True,
    counts: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the pair distribution function (PDF) of a list of distances.

    This is the single normalisation primitive used by every PDF backend in the package
    (``partial_pdf``, which the dense ``Scattering`` backend and ``Coordination``'s
    automatic cutoff both go through, and ``Scattering``'s cell-list backend), so that the
    definition of g(r) lives in exactly one place.

    Parameters:
        dist_list (np.ndarray): An array of distances (any shape; it is flattened).
        volume (float): The volume of the system.
        rrange (float, optional): The range of the PDF. Defaults to 10.
        nbin (int, optional): The number of bins. Defaults to 100.
        n_pairs (int, optional): Number of ordered pairs represented in ``dist_list``. This is
            the one quantity that differs between backends:

            - dense like pair (e.g. Si-Si): ``n_i * (n_i - 1)``, excluding self-pairs
            - dense cross pair (e.g. Si-O): ``n_i * n_j``
            - neighbour list: as above, but cross pairs are double-counted: ``2 * n_i * n_j``

            Defaults to ``dist_list.size``, which over-counts like pairs by ``n_i / (n_i - 1)``;
            callers should pass it explicitly.
        exclude_self (bool, optional): Zero the first bin to drop the zero-distance self-pairs
            present on the diagonal of a dense distance matrix. A neighbour list never contains
            them, and a cross-pair block has none either. Defaults to True.
        counts (np.ndarray, optional): A histogram of shape (nbin,) that has already been binned
            over (0, rrange), as produced by ``cell_list_pair_counts``. Given this, ``dist_list``
            is ignored and may be None; normalisation is otherwise identical, so a backend that
            never materialises its distances still shares this definition of g(r).

    Returns:
        xval (np.ndarray): The x values of the PDF.
        pdf (np.ndarray): The PDF values.
    """
    xval, volbin = radial_bins(rrange, nbin)
    if counts is not None:
        h = np.array(counts, dtype=np.int64)
        n_binned = int(h.sum())
    else:
        distances = np.asarray(dist_list)
        h, _ = np.histogram(distances, bins=nbin, range=(0, rrange))
        n_binned = distances.size
    if exclude_self:
        h[0] = 0
    if n_pairs is None:
        n_pairs = n_binned
    if n_pairs == 0:
        # Species absent, or a single atom of a like pair: no pairs to bin.
        return xval, np.zeros(nbin)
    return xval, (h / volbin) / (n_pairs / volume)


def partial_pdf(
    distances: np.ndarray,
    symbols: ArrayLike,
    volume: float,
    pair: Sequence[str | int],
    rrange: float = 10,
    nbin: int = 100,
    indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partial pair distribution function g_ab(r) from a precomputed distance matrix.

    Args:
        distances (np.ndarray): Symmetric N x N distance matrix, e.g. from ``distance_matrix``.
        symbols (array_like): Per-atom labels to match ``pair`` against, e.g. chemical symbols
            from ``atoms.get_chemical_symbols()`` or atomic numbers. Ignored if ``indices``
            is given.
        volume (float): The volume of the system.
        pair (Sequence): The two species to correlate, matched against ``symbols``.
        rrange (float, optional): The range of the PDF. Defaults to 10.
        nbin (int, optional): The number of bins. Defaults to 100.
        indices (Optional[Sequence[np.ndarray]], optional): Two arrays of atom indices, used
            in place of selecting on ``symbols``. Overrides ``pair``. Defaults to None.

    Returns:
        xval (np.ndarray): Bin centres, shape (nbin,).
        pdf (np.ndarray): g_ab(r), shape (nbin,).
    """
    if indices is None:
        symbols = np.asarray(symbols)
        atom_1 = np.flatnonzero(symbols == pair[0])
        atom_2 = np.flatnonzero(symbols == pair[1])
    else:
        atom_1, atom_2 = np.asarray(indices[0]), np.asarray(indices[1])

    if len(atom_1) == 0 or len(atom_2) == 0:
        # One species is absent; np.ix_ on an empty set would give an empty block anyway.
        xval, _ = radial_bins(rrange, nbin)
        return xval, np.zeros(nbin)

    # A like pair draws from the same index set, so the submatrix contains the zero-distance
    # diagonal and each atom has only n - 1 distinct partners. A cross pair has neither.
    like_pair = np.array_equal(atom_1, atom_2)
    if like_pair:
        n_pairs = len(atom_1) * (len(atom_1) - 1)
    else:
        n_pairs = len(atom_1) * len(atom_2)

    dist_list = distances[np.ix_(atom_1, atom_2)]
    return pdf(
        dist_list,
        volume,
        rrange,
        nbin,
        n_pairs=n_pairs,
        exclude_self=like_pair,
    )


@njit(parallel=True, cache=True)
def get_dist_numba_triclinic(
    pos: np.ndarray, rcell: np.ndarray, inverse: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """
    Distance matrix for a general (triclinic) cell, using Numba for performance.

    Positions need not be wrapped into the cell: rounding the fractional separation brings
    every pair within one cell before the image search starts. A cell whose axes are
    mutually perpendicular gets a single zero offset from `_minkowski_basis`, so the image
    search collapses and the kernel costs no more than an orthorhombic-only one would.

    `ase.Atoms.get_all_distances(mic=True)` computes the same thing and is one line, but it
    materialises the whole image search in numpy: measured at 4.8 s against 7 ms here for
    3,000 atoms, so this kernel stays.

    Args:
        pos (np.ndarray): Array of atomic positions (N x 3).
        rcell (np.ndarray): A Minkowski-reduced 3x3 cell matrix, from `_minkowski_basis`.
        inverse (np.ndarray): The inverse of `rcell`, mapping cartesian to fractional.
        offsets (np.ndarray): Candidate image translations (M x 3), from `_minkowski_basis`.

    Returns:
        np.ndarray: Symmetric distance matrix (N x N).
    """
    n = pos.shape[0]
    m = offsets.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in prange(n):  # type: ignore[attr-defined]
        for j in range(i + 1, n):  # Only calculate the upper triangle
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            # Into fractional coordinates, where a whole number of cells can be subtracted.
            fx = dx * inverse[0, 0] + dy * inverse[1, 0] + dz * inverse[2, 0]
            fy = dx * inverse[0, 1] + dy * inverse[1, 1] + dz * inverse[2, 1]
            fz = dx * inverse[0, 2] + dy * inverse[1, 2] + dz * inverse[2, 2]
            fx -= np.rint(fx)
            fy -= np.rint(fy)
            fz -= np.rint(fz)

            # Back to cartesian. This is the shortest separation only for a cell whose axes
            # are near-perpendicular, so the neighbouring images are searched below.
            bx = fx * rcell[0, 0] + fy * rcell[1, 0] + fz * rcell[2, 0]
            by = fx * rcell[0, 1] + fy * rcell[1, 1] + fz * rcell[2, 1]
            bz = fx * rcell[0, 2] + fy * rcell[1, 2] + fz * rcell[2, 2]

            best = np.inf
            for k in range(m):
                ex = bx + offsets[k, 0]
                ey = by + offsets[k, 1]
                ez = bz + offsets[k, 2]
                squared = ex * ex + ey * ey + ez * ez
                if squared < best:
                    best = squared

            d = np.sqrt(best)

            # Fill both symmetric entries
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def _minkowski_basis(cell: ArrayLike, pbc: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A Minkowski-reduced basis, its inverse, and the image translations to search alongside it.

    Rounding the fractional separation does not by itself give the minimum image in a
    skewed cell: the nearest periodic image can sit in a diagonal neighbour. On a
    Minkowski-reduced basis the neighbouring images are enough to cover that, which is the
    same guarantee `ase.geometry.find_mic` rests on. Reduction also keeps the cell-list grid
    from being badly distorted, and makes a relabelled cell (say b -> a + b) bin exactly as
    the cell it is a relabelling of. Only periodic axes are translated along, so a slab or a
    cluster is not wrapped through its free directions.

    Args:
        cell (array_like): A 3x3 cell matrix, e.g. from ``atoms.get_cell()``.
        pbc (array_like): Per-axis periodicity, e.g. ``atoms.pbc``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The reduced 3x3 cell, its inverse, and
            the image translations, shape (M, 3), with M = 27 for a fully periodic skewed
            cell and M = 1 where no search is needed.
    """
    periodic = np.asarray(pbc, dtype=bool)
    reduced = _reduced_cell(cell, periodic)
    # With mutually perpendicular axes the squared separation splits into independent
    # per-axis terms, each already minimised by the rounding, so there is nothing to search:
    # every orthorhombic cell, and any cell reducing to one, skips the 27 images entirely.
    gram = reduced @ reduced.T
    off_diagonal = gram - np.diag(np.diagonal(gram))
    if np.all(np.abs(off_diagonal) <= 1e-10 * np.abs(np.diagonal(gram)).max()):
        offsets = np.zeros((1, 3))
    else:
        steps = [(-1, 0, 1) if p else (0,) for p in periodic]
        offsets = np.array(
            [i * reduced[0] + j * reduced[1] + k * reduced[2] for i in steps[0] for j in steps[1] for k in steps[2]]
        )
    return reduced, np.ascontiguousarray(np.linalg.inv(reduced)), offsets


@njit(parallel=True, cache=True)
def _cell_list_pair_hist(
    cart: np.ndarray,
    rcell: np.ndarray,
    codes: np.ndarray,
    cell_of: np.ndarray,
    cell_start: np.ndarray,
    n_species: int,
    ncx: int,
    ncy: int,
    ncz: int,
    gx: int,
    gy: int,
    gz: int,
    rrange: float,
    nbin: int,
    nchunks: int,
) -> np.ndarray:
    """
    Histogram of pair distances by species pair, from a cell list. See `cell_list_pair_counts`.

    Args:
        cart (np.ndarray): Cartesian positions wrapped into the cell, shape (N, 3), ordered
            by cell so that each cell's atoms are contiguous. Cartesian rather than
            fractional so the inner loop subtracts instead of applying `rcell` per pair.
        rcell (np.ndarray): The 3x3 cell, used to turn a cell-index wrap into a shift.
        codes (np.ndarray): Per-atom species code in [0, n_species), shape (N,), in the same
            order as `cart`.
        cell_of (np.ndarray): Each atom's (i, j, k) cell, shape (N, 3), same order again.
        cell_start (np.ndarray): Offsets into `cart` of each cell's block, shape
            (ncx * ncy * ncz + 1,), so cell c holds `cell_start[c] : cell_start[c + 1]`.
        n_species (int): The number of species.
        ncx, ncy, ncz (int): Cells along each axis, each at least 1.
        gx, gy, gz (int): Stencil radius in cells along each axis: at least the number of
            cells `rrange` spans, so that every neighbour is inside the stencil. May exceed
            the corresponding `nc`, in which case the stencil wraps onto itself.
        rrange (float): The cutoff, equal to the top of the histogram range.
        nbin (int): The number of bins.
        nchunks (int): How many strided slices to split the atoms into, one per parallel
            iteration and one private histogram each. Any positive value is correct;
            the thread count is the value that balances.

    Returns:
        np.ndarray: Counts of shape (n_species * n_species, nbin), int64, with each unordered
            pair counted twice -- once per direction. The key is symmetric in the two species,
            so the two directions land in the same row and bin.
    """
    n = cart.shape[0]
    hist = np.zeros((nchunks, n_species * n_species, nbin), dtype=np.int64)
    r2max = rrange * rrange
    scale = nbin / rrange

    for tid in prange(nchunks):  # type: ignore[attr-defined]
        for i in range(tid, n, nchunks):
            code_i = codes[i]
            x0 = cart[i, 0]
            y0 = cart[i, 1]
            z0 = cart[i, 2]
            for sx in range(gx + 1):
                rx = cell_of[i, 0] + sx
                wx = rx // ncx
                rx -= wx * ncx
                for sy in range(-gy if sx > 0 else 0, gy + 1):
                    ry = cell_of[i, 1] + sy
                    wy = ry // ncy
                    ry -= wy * ncy
                    for sz in range(-gz if (sx > 0 or sy > 0) else 0, gz + 1):
                        rz = cell_of[i, 2] + sz
                        wz = rz // ncz
                        rz -= wz * ncz
                        ox = wx * rcell[0, 0] + wy * rcell[1, 0] + wz * rcell[2, 0]
                        oy = wx * rcell[0, 1] + wy * rcell[1, 1] + wz * rcell[2, 1]
                        oz = wx * rcell[0, 2] + wy * rcell[1, 2] + wz * rcell[2, 2]

                        c = (rx * ncy + ry) * ncz + rz
                        if sx == 0 and sy == 0 and sz == 0:
                            j0 = i + 1
                        else:
                            j0 = cell_start[c]
                        for j in range(j0, cell_start[c + 1]):
                            if j == i:
                                continue
                            dx = cart[j, 0] + ox - x0
                            dy = cart[j, 1] + oy - y0
                            dz = cart[j, 2] + oz - z0
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < r2max:
                                b = int(np.sqrt(d2) * scale)
                                if b >= nbin:
                                    b = nbin - 1
                                code_j = codes[j]
                                key = min(code_i, code_j) * n_species + max(code_i, code_j)
                                hist[tid, key, b] += 1
    return 2 * hist.sum(axis=0)


def cell_list_pair_counts(atoms: Atoms, codes: np.ndarray, rrange: float, nbin: int, n_species: int) -> np.ndarray:
    """
    Binned pair distances within `rrange`, by species pair, without building a pair list.

    A cell list sorts the atoms into boxes, so each atom only has to be compared with the
    half-block of boxes around it that `rrange` can reach -- half, because the other half
    finds the same pairs from the far end. Distances go straight into the
    histogram, so memory is linear in N, where a materialised pair list costs one record per
    pair. Time is linear in N times the neighbours per atom, so linear in N only while the box
    stays large relative to `rrange`; at `rrange` near the minimum image limit the sphere holds
    a fixed fraction of the cell and the work is quadratic whatever the algorithm. Each
    unordered pair is reported twice, matching `ase.neighborlist.neighbor_list`, so callers
    keep the same `n_pairs` normalisation.

    Requires a cell periodic along all three axes, and `rrange` at most half the shortest
    perpendicular width -- the minimum image limit, past which the counts would describe
    periodic replicas rather than neighbours.

    Args:
        atoms (Atoms): An ASE Atoms object. Positions need not be wrapped into the cell.
        codes (np.ndarray): Per-atom species code in [0, n_species), shape (N,).
        rrange (float): The cutoff, equal to the top of the histogram range.
        nbin (int): The number of bins.
        n_species (int): The number of species.

    Returns:
        np.ndarray: Counts of shape (n_species * n_species, nbin), indexed by
            ``min(code_i, code_j) * n_species + max(code_i, code_j)`` so that both orderings
            of a cross pair share a row. Each unordered pair is counted twice, once per
            direction, which is the convention ``pdf``'s ``n_pairs`` is built on:
            ``2 * n_i * n_j`` for a cross pair, ``n_i * (n_i - 1)`` for a like pair.

    Raises:
        ValueError: If the cell is not periodic along all three axes, or `rrange` exceeds
            half the shortest perpendicular width.
    """
    codes = np.asarray(codes)
    if codes.shape != (len(atoms),):
        raise ValueError(f"codes must have one entry per atom, but got shape {codes.shape} for {len(atoms)} atoms.")
    if codes.size and not 0 <= codes.min() <= codes.max() < n_species:
        raise ValueError(
            f"codes must lie in [0, n_species) = [0, {n_species}), but they run from {codes.min()} to {codes.max()}."
        )

    cell = atoms.get_cell()
    if not np.asarray(atoms.pbc).all():
        raise ValueError(
            f"A cell periodic along all three axes is required, but got pbc={atoms.pbc}. "
            "The minimum image convention is applied unconditionally, so a free surface would "
            "be silently folded in rather than left alone. For an isolated cluster, pad the box "
            "to at least twice the cluster's extent and set pbc=True."
        )

    rcell, inverse, _ = _minkowski_basis(cell, atoms.pbc)
    widths = perpendicular_widths(rcell)

    # == minimum_image_limit(cell, atoms.pbc), from the reduction already in hand.
    limit = min(perpendicular_widths(cell).min(), widths.min()) / 2
    if rrange > limit:
        raise ValueError(
            f"rrange ({rrange:.2f} A) exceeds the minimum image limit ({limit:.2f} A), beyond "
            "which the counts describe periodic images rather than real neighbours. The limit "
            "is the narrower of this cell and its Minkowski reduction; see "
            "`vitrum.geometry.minimum_image_limit`."
        )

    nc = np.maximum(np.floor(_GRID_REFINE * widths / rrange), 1).astype(np.int64)
    stencil = np.ceil(rrange * nc / widths).astype(np.int64)

    frac = np.asarray(atoms.get_positions(), dtype=float) @ inverse
    frac -= np.floor(frac)

    ncell = int(nc[0] * nc[1] * nc[2])
    cell_of = np.minimum((frac * nc).astype(np.int64), nc - 1)
    cell_id = (cell_of[:, 0] * nc[1] + cell_of[:, 1]) * nc[2] + cell_of[:, 2]
    order = np.argsort(cell_id, kind="stable")
    cell_start = np.zeros(ncell + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell_id, minlength=ncell), out=cell_start[1:])

    # The kernel works in Cartesian coordinates, so that a pair separation is a subtraction
    # rather than a 3x3 product per pair. Only the wrap shifts still go through `rcell`, and
    # there is one of those per cell rather than one per pair.
    cart = frac @ rcell

    return _cell_list_pair_hist(
        np.ascontiguousarray(cart[order]),
        rcell,
        np.ascontiguousarray(codes[order], dtype=np.int64),
        np.ascontiguousarray(cell_of[order]),
        cell_start,
        int(n_species),
        int(nc[0]),
        int(nc[1]),
        int(nc[2]),
        int(stencil[0]),
        int(stencil[1]),
        int(stencil[2]),
        float(rrange),
        int(nbin),
        int(get_num_threads()),
    )


def distance_matrix(atoms: Atoms) -> np.ndarray:
    """
    Minimum-image distance matrix between all pairs of atoms in a structure.
    Positions need not be wrapped into the cell.

    Any cell is accepted. The cell is reduced to a Minkowski basis and the nearest periodic
    image searched for, which a cell with perpendicular axes skips entirely -- there the
    rounding already gives the minimum image, so it costs no more than an
    orthorhombic-only kernel would.

    Args:
        atoms (Atoms): An ASE Atoms object.

    Returns:
        np.ndarray: Symmetric distance matrix (N x N).
    """
    positions = np.ascontiguousarray(atoms.get_positions(), dtype=float)
    return get_dist_numba_triclinic(positions, *_minkowski_basis(atoms.get_cell(), atoms.pbc))


def get_dist(list: np.ndarray, cell: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise distance matrix for atoms in a periodic simulation box.

    Kept as a plain-numpy (non-numba) reference implementation for backward
    compatibility; `distance_matrix` is what the analysis classes use, and it takes an
    `Atoms` object rather than positions and a cell diagonal. Unlike `distance_matrix`,
    this takes a cell diagonal only and so remains orthorhombic-only.

    Positions need not be wrapped into the cell: the minimum image is taken by
    subtracting a whole number of cell lengths.

    Args:
        list (np.ndarray): Atomic positions (N x 3).
        cell (np.ndarray): Cell dimensions (3,). Assumes an orthorhombic cell.

    Returns:
        np.ndarray: Symmetric distance matrix (N x N) containing distances between all atom pairs.
    """
    dim = np.asarray([cell[0], cell[1], cell[2]], dtype=float)
    diff = np.asarray(list, dtype=float)[np.newaxis, :, :] - np.asarray(list, dtype=float)[:, np.newaxis, :]
    lengths = np.where(dim > 0.0, dim, 1.0)
    diff -= np.where(dim > 0.0, lengths * np.rint(diff / lengths), 0.0)
    return np.sqrt(np.sum(diff**2, axis=-1))
