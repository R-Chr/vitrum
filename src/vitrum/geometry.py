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


def find_min_after_peak(padf):
    """
    Find the index of the first local minimum after the first peak in a function.
    Useful for determining cutoffs from PDFs.

    Args:
        padf (np.ndarray): The probability density function or similar array.

    Returns:
        int: The index of the minimum.
    """
    mins = argrelextrema(padf, np.less_equal, order=4)[0]
    second_min = [i for ind, i in enumerate(mins) if i != ind][0]
    return second_min


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
    (``GlassAtoms.get_pdf`` and both ``Scattering`` backends), so that the definition of
    g(r) lives in exactly one place.

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


@njit(parallel=True)
def get_dist_numba(pos, cell):
    """
    Calculate the distance matrix between atoms using Numba for performance.

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
    half_lx, half_ly, half_lz = lx / 2.0, ly / 2.0, lz / 2.0

    for i in prange(n):
        for j in range(i + 1, n):  # Only calculate the upper triangle
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            dz = abs(pos[i, 2] - pos[j, 2])

            # Apply Periodic Boundary Conditions (Minimum Image Convention)
            if dx > half_lx: dx -= lx
            if dy > half_ly: dy -= ly
            if dz > half_lz: dz -= lz

            d = np.sqrt(dx**2 + dy**2 + dz**2)

            # Fill both symmetric entries
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def get_dist(list, cell):
    """
    Calculate the pairwise distance matrix for atoms in a periodic simulation box.

    Kept as a plain-numpy (non-numba) reference implementation for backward
    compatibility; get_dist_numba is the version used internally by GlassAtoms.

    Args:
        list (np.ndarray): Atomic positions (N x 3).
        cell (np.ndarray): Cell dimensions (3,). Assumes an orthorhombic cell.

    Returns:
        np.ndarray: Symmetric distance matrix (N x N) containing distances between all atom pairs.
    """
    dim = [cell[0], cell[1], cell[2]]
    x_dif = np.abs(list[:, 0][np.newaxis, :] - list[:, 0][:, np.newaxis])
    y_dif = np.abs(list[:, 1][np.newaxis, :] - list[:, 1][:, np.newaxis])
    z_dif = np.abs(list[:, 2][np.newaxis, :] - list[:, 2][:, np.newaxis])
    x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
    y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
    z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
    return np.sqrt(x_dif**2 + y_dif**2 + z_dif**2)
