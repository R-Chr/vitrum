import numpy as np
from numba import njit, prange
from scipy.signal import argrelextrema


def require_orthorhombic(cell, caller: str = "") -> np.ndarray:
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


def find_min_after_peak(padf, context: str = ""):
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


def _crossing(x, y, start: int, level: float, step: int) -> float | None:
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


def radial_bins(rrange=10, nbin=100):
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


def pdf(dist_list, volume, rrange=10, nbin=100, n_pairs=None, exclude_self=True):
    """
    Calculate the pair distribution function (PDF) of a list of distances.

    This is the single normalisation primitive used by every PDF backend in the package
    (``partial_pdf``, which the dense ``Scattering`` backend and ``Coordination``'s
    automatic cutoff both go through, and ``Scattering``'s neighbour-list backend), so that
    the definition of g(r) lives in exactly one place.

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

    Returns:
        xval (np.ndarray): The x values of the PDF.
        pdf (np.ndarray): The PDF values.
    """
    dist_list = np.asarray(dist_list)
    xval, volbin = radial_bins(rrange, nbin)
    h, _ = np.histogram(dist_list, bins=nbin, range=(0, rrange))
    if exclude_self:
        h[0] = 0
    if n_pairs is None:
        n_pairs = dist_list.size
    if n_pairs == 0:
        # Species absent, or a single atom of a like pair: no pairs to bin.
        return xval, np.zeros(nbin)
    return xval, (h / volbin) / (n_pairs / volume)


def partial_pdf(distances, symbols, volume, pair, rrange=10, nbin=100, indices=None):
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


@njit(parallel=True)
def get_dist_numba(pos, cell):
    """
    Calculate the distance matrix between atoms using Numba for performance.

    Positions need not be wrapped into the cell: the minimum image is taken by
    subtracting a whole number of cell lengths.

    Args:
        pos (np.ndarray): Array of atomic positions (N x 3).
        cell (np.ndarray): Cell dimensions (3,). Assumes an orthorhombic cell (lx, ly, lz).

    Returns:
        np.ndarray: Symmetric distance matrix (N x N).
    """
    n = pos.shape[0]
    # Initialize the output matrix
    dist_matrix = np.zeros((n, n))

    # Extract cell dimensions for faster access
    lx, ly, lz = cell[0], cell[1], cell[2]

    for i in prange(n):
        for j in range(i + 1, n):  # Only calculate the upper triangle
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            # Apply Periodic Boundary Conditions (Minimum Image Convention)
            if lx > 0.0:
                dx -= lx * np.rint(dx / lx)
            if ly > 0.0:
                dy -= ly * np.rint(dy / ly)
            if lz > 0.0:
                dz -= lz * np.rint(dz / lz)

            d = np.sqrt(dx**2 + dy**2 + dz**2)

            # Fill both symmetric entries
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def distance_matrix(atoms, caller: str = "") -> np.ndarray:
    """
    Minimum-image distance matrix between all pairs of atoms in a structure.
    Positions need not be wrapped into the cell.

    Args:
        atoms (Atoms): An ASE Atoms object.
        caller (str, optional): Name of the calling routine, used in the error message.

    Returns:
        np.ndarray: Symmetric distance matrix (N x N).

    Raises:
        NotImplementedError: If the cell is not orthorhombic.
    """
    dim = require_orthorhombic(atoms.get_cell(), caller or "distance_matrix")
    return get_dist_numba(atoms.get_positions(), dim)


def get_dist(list, cell):
    """
    Calculate the pairwise distance matrix for atoms in a periodic simulation box.

    Kept as a plain-numpy (non-numba) reference implementation for backward
    compatibility; `distance_matrix` is what the analysis classes use, and it takes an
    `Atoms` object rather than positions and a cell diagonal.

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
