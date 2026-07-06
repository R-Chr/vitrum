from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import write
from ase.symbols import symbols2numbers
from scipy.ndimage import label as ndi_label
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from vitrum.glass_atoms import GlassAtoms


def compute_occupancy_grid(
    atoms: Atoms,
    grid_spacing: float = 0.2,
    radii_scaling: float = 1.0,
    probe_radius: float = 0.0,
    radii_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Classify a fine 3D grid over the (orthorhombic) cell as occupied or free.

    A grid point is "occupied" if it lies within some atom's exclusion radius
    (covalent radius, scaled and offset by a probe radius), and "free" otherwise.
    Periodicity is handled natively via `scipy.spatial.cKDTree`'s `boxsize` support.

    Args:
        atoms (ase.Atoms): The structure to analyze. Must have an orthorhombic cell.
        grid_spacing (float): Target spacing between grid points, in Angstrom. The
            actual spacing used is rounded per axis so the grid evenly tiles the
            cell; see the returned `spacing`. Defaults to 0.2.
        radii_scaling (float): Factor multiplying each atom's exclusion radius.
            Defaults to 1.0.
        probe_radius (float): Additive probe radius, in Angstrom, added to every
            atom's (scaled) exclusion radius. Defaults to 0.0.
        radii_overrides (Optional[Dict[str, float]]): Per-species base radius (in
            Angstrom, before `radii_scaling` is applied) to use instead of
            `ase.data.covalent_radii`. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - occupied: Boolean array of shape (nx, ny, nz). True where the grid
              point falls inside an atom's exclusion sphere.
            - spacing: Array of shape (3,), the actual per-axis grid spacing used.
    '''
    atoms = atoms.copy()
    atoms.wrap()
    cell_lengths = np.diagonal(atoms.get_cell())

    symbols = np.array(atoms.get_chemical_symbols())
    base_radii = covalent_radii[symbols2numbers(symbols)].astype(float)
    if radii_overrides:
        for symbol, radius in radii_overrides.items():
            base_radii[symbols == symbol] = radius
    final_radii = base_radii * radii_scaling + probe_radius

    if np.any(final_radii > cell_lengths.min() / 2):
        print(
            "WARNING: at least one exclusion radius exceeds half the smallest cell "
            "length; periodic nearest-neighbor queries may be ambiguous."
        )

    n = np.maximum(1, np.round(cell_lengths / grid_spacing).astype(int))
    spacing = cell_lengths / n

    if np.prod(n, dtype=np.int64) > 20_000_000:
        print(
            f"WARNING: occupancy grid has {np.prod(n, dtype=np.int64):,} points "
            f"(grid_spacing={grid_spacing}); consider a coarser grid_spacing for "
            "faster analysis."
        )

    axes = [np.arange(n[i]) * spacing[i] for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=1)

    positions = atoms.get_positions()
    occupied = np.zeros(grid_points.shape[0], dtype=bool)
    for r in np.unique(final_radii):
        tree = cKDTree(positions[final_radii == r], boxsize=cell_lengths)
        dist, _ = tree.query(grid_points, k=1)
        occupied |= dist <= r

    return occupied.reshape(n[0], n[1], n[2]), spacing


def find_cavities(
    occupied: np.ndarray,
    min_grid_points: int = 4,
) -> List[np.ndarray]:
    '''
    Cluster free (non-occupied) grid points into discrete, periodicity-aware cavities.

    Connectivity is face-adjacency only (6-connected): two free voxels that only
    touch at an edge or corner represent a zero-width channel and are not considered
    connected. `scipy.ndimage.label` has no notion of periodicity, so labels that
    touch across opposite cell faces are merged in a second pass.

    Args:
        occupied (np.ndarray): Boolean array of shape (nx, ny, nz), from
            `compute_occupancy_grid`.
        min_grid_points (int): Minimum number of voxels for a cluster to be kept as
            a cavity; smaller clusters are grid-quantization noise. Defaults to 4.

    Returns:
        List[np.ndarray]: One array of shape (n, 3) per cavity, the (ix, iy, iz)
            grid indices of its free voxels, already merged across periodic
            boundaries.
    '''
    free = ~occupied
    labels, n_labels = ndi_label(free)

    edges = set()
    for axis in range(3):
        first = np.take(labels, 0, axis=axis)
        last = np.take(labels, -1, axis=axis)
        mask = (first > 0) & (last > 0)
        if np.any(mask):
            pairs = np.unique(np.stack([first[mask], last[mask]], axis=1), axis=0)
            edges.update(map(tuple, pairs))

    if edges:
        idx = np.array(list(edges)).T
        adj = csr_array(
            (np.ones(idx.shape[1]), (idx[0], idx[1])),
            shape=(n_labels + 1, n_labels + 1),
        )
        _, component_id = connected_components(adj, directed=False)
        merged = np.zeros_like(labels)
        nonzero = labels > 0
        merged[nonzero] = component_id[labels[nonzero]]
        labels = merged

    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    surviving = unique_labels[counts >= min_grid_points]

    return [np.argwhere(labels == lbl) for lbl in surviving]


def _cell_edges(cell_lengths: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Pairs of (start, stop) Cartesian points for the 12 edges of an axis-aligned box.
    '''
    org = np.zeros(3)
    a = np.array([cell_lengths[0], 0, 0])
    b = np.array([0, cell_lengths[1], 0])
    c = np.array([0, 0, cell_lengths[2]])
    abc = a + b + c
    return [
        (org, a), (org, b), (org, c),
        (a, a + b), (a, a + c),
        (b, b + a), (b, b + c),
        (c, c + a), (c, c + b),
        (abc, abc - a), (abc, abc - b), (abc, abc - c),
    ]


def _add_cell_edges_trace(fig, cell_lengths: np.ndarray) -> None:
    '''Add the 12 edges of the (orthorhombic) simulation cell to a plotly figure.'''
    import plotly.graph_objects as go

    for start, stop in _cell_edges(cell_lengths):
        fig.add_trace(
            go.Scatter3d(
                x=(start[0], stop[0]),
                y=(start[1], stop[1]),
                z=(start[2], stop[2]),
                mode="lines",
                line={"color": "black", "width": 2},
                showlegend=False,
            )
        )


def _add_atoms_trace(
    fig,
    atoms: Atoms,
    colors: Optional[Dict[str, str]] = None,
    marker_size: int = 4,
) -> None:
    '''Add one Scatter3d marker trace per chemical species to a plotly figure.'''
    import plotly.express as px
    import plotly.graph_objects as go

    symbols = np.array(atoms.get_chemical_symbols())
    species = np.unique(symbols)

    if colors is None:
        palette = px.colors.qualitative.Plotly
        colors = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    positions = atoms.get_positions()
    for s in species:
        pts = positions[symbols == s]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=s,
                marker={"size": marker_size, "color": colors.get(s)},
            )
        )


def _add_void_isosurface_trace(
    fig,
    occupied: np.ndarray,
    spacing: np.ndarray,
    isovalue: float = 0.5,
    smoothing_sigma: float = 1.0,
    color: str = "cyan",
    opacity: float = 0.35,
) -> None:
    '''
    Add a marching-cubes isosurface mesh of the free (void) space to a plotly figure.

    The free/occupied grid is optionally Gaussian-smoothed (with periodic wrapping,
    to avoid a seam at the cell boundary) before extracting the isosurface, since
    the raw voxel grid produces a blocky surface.
    '''
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    from skimage import measure

    free = (~occupied).astype(float)
    if smoothing_sigma > 0:
        free = gaussian_filter(free, sigma=smoothing_sigma, mode="wrap")

    if free.min() >= isovalue or free.max() <= isovalue:
        return

    verts, faces, *_ = measure.marching_cubes(free, level=isovalue)
    cart_verts = verts * spacing

    fig.add_trace(
        go.Mesh3d(
            x=cart_verts[:, 0],
            y=cart_verts[:, 1],
            z=cart_verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name="void space",
            showlegend=True,
        )
    )


class Cavity(object):
    """
    A discrete free-volume region ("void") in an atomistic system.
    """

    def __init__(
        self,
        atoms: Atoms,
        voxel_indices: np.ndarray,
        spacing: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ):
        """
        Initialize a Cavity object.

        Args:
            atoms (Atoms): The parent structure (used only for cell/pbc information;
                a Cavity does not "contain" atoms the way a Ring does).
            voxel_indices (np.ndarray): Array of shape (n, 3), the (ix, iy, iz) grid
                indices of the free voxels belonging to this cavity.
            spacing (np.ndarray): Array of shape (3,), the grid spacing used to
                produce `voxel_indices` (from `compute_occupancy_grid`).
            grid_shape (Tuple[int, int, int]): The (nx, ny, nz) shape of the full
                occupancy grid this cavity was extracted from.
        """
        self.atoms = atoms
        self.voxel_indices = voxel_indices
        self.spacing = spacing
        self.grid_shape = grid_shape
        self._center_cache = None

    def n_grid_points(self) -> int:
        """
        Number of free grid points belonging to this cavity.

        Returns:
            int: The number of grid points.
        """
        return len(self.voxel_indices)

    def volume(self) -> float:
        """
        Calculate the volume of the cavity.

        Returns:
            float: The volume of the cavity, in Angstrom^3.
        """
        return self.n_grid_points() * float(np.prod(self.spacing))

    def effective_radius(self) -> float:
        """
        Calculate the radius of a sphere with the same volume as the cavity.

        Returns:
            float: The effective radius of the cavity, in Angstrom.
        """
        return float((3 * self.volume() / (4 * np.pi)) ** (1 / 3))

    def fractional_center(self) -> np.ndarray:
        """
        Calculate the periodic-safe centroid of the cavity, in fractional coordinates.

        Each axis is periodic, so the centroid is computed as a circular mean of the
        voxels' fractional coordinates on that axis (treating the fractional
        coordinate as an angle) rather than a naive arithmetic mean, which would be
        wrong for a cavity that straddles a cell boundary.

        Note:
            If a cavity's voxels are spread near-uniformly across most or all of a
            periodic axis (e.g. a percolating/channel-like cavity), the circular
            mean becomes numerically ill-conditioned and the result is not
            physically meaningful. `volume()`, `effective_radius()`, and
            `n_grid_points()` are unaffected by this, since they only count voxels.
            Use `spans_full_cell()` to detect this case.

        Returns:
            np.ndarray: Array of shape (3,), fractional coordinates in [0, 1).
        """
        if self._center_cache is None:
            frac = (self.voxel_indices + 0.5) / np.array(self.grid_shape)
            theta = 2 * np.pi * frac
            theta_mean = np.arctan2(np.sin(theta).mean(axis=0), np.cos(theta).mean(axis=0))
            self._center_cache = (theta_mean / (2 * np.pi)) % 1.0
        return self._center_cache

    def center(self) -> np.ndarray:
        """
        Calculate the periodic-safe centroid of the cavity, in Cartesian coordinates.

        Returns:
            np.ndarray: Array of shape (3,), the center of the cavity, wrapped into
                the primary cell. See `fractional_center` for the limitations of
                this calculation for percolating cavities.
        """
        cell = self.atoms.get_cell()
        return cell.cartesian_positions(self.fractional_center())

    def spans_full_cell(self, axis: Optional[int] = None) -> bool:
        """
        Check whether the cavity's voxels cover every grid index along an axis.

        Useful as a cheap diagnostic for whether `center()` is trustworthy: a
        cavity that fully spans a periodic axis is percolating/channel-like, and
        its centroid along that axis is not physically meaningful.

        Args:
            axis (Optional[int]): Which axis (0, 1, or 2) to check. If None, checks
                all three axes and returns True if any of them is fully spanned.

        Returns:
            bool: True if the cavity covers every grid index along the checked
                axis (axes).
        """
        axes = [axis] if axis is not None else [0, 1, 2]
        for ax in axes:
            if len(np.unique(self.voxel_indices[:, ax])) == self.grid_shape[ax]:
                return True
        return False


class VoidAnalysis:
    """
    A class for calculating and analyzing free volume and voids in atomistic systems.
    """

    def __init__(
        self,
        atoms: Atoms,
        radii_scaling: float = 1.0,
        radii_overrides: Optional[Dict[str, float]] = None,
        probe_radius: float = 0.0,
    ):
        """
        Initialize the VoidAnalysis class.

        Args:
            atoms (Atoms): The structure to analyze (a single snapshot, not a
                trajectory). Must have an orthorhombic cell.
            radii_scaling (float): Factor multiplying each atom's exclusion radius.
                Defaults to 1.0.
            radii_overrides (Optional[Dict[str, float]]): Per-species base radius
                (in Angstrom, before `radii_scaling` is applied) to use instead of
                `ase.data.covalent_radii`. Defaults to None.
            probe_radius (float): Additive probe radius, in Angstrom, added to
                every atom's (scaled) exclusion radius. Defaults to 0.0.

        Raises:
            ValueError: If the cell is not orthorhombic.
        """
        self.atoms = GlassAtoms(atoms.copy())
        self.atoms.wrap()
        self._validate_orthorhombic()
        self.radii_scaling = radii_scaling
        self.radii_overrides = radii_overrides
        self.probe_radius = probe_radius
        self.cavities: Optional[List[Cavity]] = None
        self._occupied: Optional[np.ndarray] = None
        self._spacing: Optional[np.ndarray] = None

    def _validate_orthorhombic(self):
        cell = np.array(self.atoms.get_cell())
        off_diag = cell - np.diag(np.diagonal(cell))
        if not np.allclose(off_diag, 0.0, atol=1e-8):
            raise ValueError(
                "VoidAnalysis requires an orthorhombic cell; got a cell with "
                "non-zero off-diagonal components."
            )

    def calculate(
        self,
        grid_spacing: float = 0.2,
        min_grid_points: int = 4,
    ) -> List[Cavity]:
        """
        Compute the occupancy grid and cluster it into cavities.

        Args:
            grid_spacing (float): Target spacing between grid points, in Angstrom.
                Defaults to 0.2.
            min_grid_points (int): Minimum number of voxels for a cluster to be
                kept as a cavity. Defaults to 4.

        Returns:
            List[Cavity]: A list of Cavity objects representing the discrete voids
                in the system.
        """
        occupied, spacing = compute_occupancy_grid(
            self.atoms,
            grid_spacing=grid_spacing,
            radii_scaling=self.radii_scaling,
            probe_radius=self.probe_radius,
            radii_overrides=self.radii_overrides,
        )
        self._occupied = occupied
        self._spacing = spacing

        voxel_clusters = find_cavities(occupied, min_grid_points=min_grid_points)
        self.cavities = [
            Cavity(self.atoms, voxels, spacing, occupied.shape) for voxels in voxel_clusters
        ]
        return self.cavities

    def get_free_volume_fraction(self) -> float:
        """
        Calculate the fraction of the cell volume that is free (non-occupied) space.

        Returns:
            float: The free volume fraction, between 0 and 1.

        Raises:
            ValueError: If `calculate()` has not been run yet.
        """
        if self._occupied is None:
            raise ValueError("Occupancy grid has not been calculated yet. Run .calculate() first.")
        return float(np.count_nonzero(~self._occupied) / self._occupied.size)

    def get_cavity_size_distribution(
        self, by: str = "volume", n_bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a histogram of cavity sizes.

        Note this returns a continuous histogram (bin edges and counts), unlike
        `RingAnalysis.get_ring_size_distribution`'s integer-keyed dict, since cavity
        volumes and effective radii are continuous quantities rather than small
        integers.

        Args:
            by (str): Which quantity to histogram, one of "volume" or
                "effective_radius". Defaults to "volume".
            n_bins (int): Number of histogram bins. Defaults to 20.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (bin_edges, counts) as returned by
                `np.histogram`.

        Raises:
            ValueError: If cavities have not been calculated yet, or `by` is not
                one of "volume" or "effective_radius".
        """
        if self.cavities is None:
            raise ValueError("Cavities have not been calculated yet. Run .calculate() first.")
        if by == "volume":
            sizes = [c.volume() for c in self.cavities]
        elif by == "effective_radius":
            sizes = [c.effective_radius() for c in self.cavities]
        else:
            raise ValueError(f"Unknown value for `by`: '{by}', expected 'volume' or 'effective_radius'.")

        counts, bin_edges = np.histogram(sizes, bins=n_bins)
        return bin_edges, counts

    def plot_cavity_size_distribution(self, by: str = "volume", n_bins: int = 20, ax=None, **plot_kwargs):
        """
        Plot the distribution of cavity sizes using matplotlib.

        Args:
            by (str): Which quantity to histogram, one of "volume" or
                "effective_radius". Defaults to "volume".
            n_bins (int): Number of histogram bins. Defaults to 20.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure and axes are created.
            **plot_kwargs: Additional keyword arguments passed to `ax.bar`.

        Returns:
            matplotlib.axes.Axes: The axes the distribution was plotted on.
        """
        import matplotlib.pyplot as plt

        if not self.cavities:
            print("No cavities found. Ensure you have run .calculate() first.")
            return ax

        bin_edges, counts = self.get_cavity_size_distribution(by=by, n_bins=n_bins)
        centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        widths = np.diff(bin_edges)

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
            xlabel = "Cavity volume (Å$^3$)" if by == "volume" else "Cavity effective radius (Å)"
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("N$_{cavities}$", fontsize=12)

        ax.bar(centers, counts, width=widths, **plot_kwargs)

        return ax

    def plot_3d(
        self,
        fig=None,
        isovalue: float = 0.5,
        smoothing_sigma: float = 1.0,
        show_atoms: bool = True,
        atom_colors: Optional[Dict[str, str]] = None,
        atom_marker_size: int = 4,
        void_color: str = "cyan",
        void_opacity: float = 0.35,
        title: str = "Void analysis",
    ):
        """
        Build an interactive 3D plot of the void space together with the atoms.

        The void space is rendered as a marching-cubes isosurface mesh over the
        occupancy grid, the atoms as per-species marker traces, and the simulation
        cell as a wireframe box, following the plotly 3D plotting approach of
        GEMDAT (https://github.com/GEMDAT-repos/GEMDAT/blob/main/src/gemdat/plots/plotly/_plot3d.py),
        adapted here for a binary occupancy grid instead of a continuous
        probability density.

        Requires the optional `plotly` and `scikit-image` dependencies — see
        Installation docs.

        Args:
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
                If None, a new figure is created.
            isovalue (float): Isosurface threshold on the (optionally smoothed)
                free-space grid, between 0 (occupied) and 1 (free). Defaults to 0.5.
            smoothing_sigma (float): Gaussian smoothing sigma, in grid points,
                applied (with periodic wrapping) to the free-space grid before
                extracting the isosurface, to avoid blocky voxel artifacts. Set to
                0 to disable. Defaults to 1.0.
            show_atoms (bool): Whether to plot the atoms as markers. Defaults to
                True.
            atom_colors (Optional[Dict[str, str]]): Mapping of chemical symbol to
                a plotly color. Defaults to None (auto-assigned).
            atom_marker_size (int): Marker size for atoms. Defaults to 4.
            void_color (str): Color of the void isosurface. Defaults to "cyan".
            void_opacity (float): Opacity of the void isosurface. Defaults to 0.35.
            title (str): Plot title. Defaults to "Void analysis".

        Returns:
            plotly.graph_objects.Figure: The resulting figure.

        Raises:
            ValueError: If the occupancy grid has not been calculated yet.

        Note:
            Marching cubes does not stitch the isosurface mesh across periodic
            cell boundaries, so a cavity that wraps the cell (see
            `Cavity.spans_full_cell`) will appear visually "cut" at the box faces
            even though it is a single connected void.
        """
        if self._occupied is None:
            raise ValueError("Occupancy grid has not been calculated yet. Run .calculate() first.")

        import plotly.graph_objects as go

        if fig is None:
            fig = go.Figure()

        cell_lengths = np.diagonal(self.atoms.get_cell())
        _add_cell_edges_trace(fig, cell_lengths)

        if show_atoms:
            _add_atoms_trace(fig, self.atoms, colors=atom_colors, marker_size=atom_marker_size)

        _add_void_isosurface_trace(
            fig,
            self._occupied,
            self._spacing,
            isovalue=isovalue,
            smoothing_sigma=smoothing_sigma,
            color=void_color,
            opacity=void_opacity,
        )

        ratio = cell_lengths / cell_lengths.max()
        fig.update_layout(
            title=title,
            scene={
                "aspectmode": "manual",
                "aspectratio": {"x": ratio[0], "y": ratio[1], "z": ratio[2]},
                "xaxis_title": "X (Å)",
                "yaxis_title": "Y (Å)",
                "zaxis_title": "Z (Å)",
            },
            legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left", "x": 0, "y": -0.1},
            showlegend=True,
            margin={"l": 0, "r": 0, "b": 0, "t": 30},
            scene_camera={"projection": {"type": "orthographic"}},
        )

        return fig

    def write_cavities(self, filename: str, format: str = "extxyz", dummy_symbol: str = "X"):
        """
        Write the structure together with a dummy pseudo-atom at each cavity center.

        This produces a single, already cell-aligned file that can be opened
        directly in a molecular viewer (e.g. ASE's GUI, OVITO, VMD) to visualize
        cavity locations alongside the real structure.

        Args:
            filename (str): The name of the file to write to.
            format (str): The file format, passed to `ase.io.write`. Defaults to
                'extxyz'.
            dummy_symbol (str): The chemical symbol used for the cavity-center
                pseudo-atoms. Defaults to "X", ASE's placeholder species, which is
                unlikely to be auto-bonded by a viewer's default bonding rules.

        Raises:
            ValueError: If cavities have not been calculated yet.
        """
        if self.cavities is None:
            raise ValueError("Cavities have not been calculated yet. Run .calculate() first.")

        centers = np.array([c.center() for c in self.cavities]).reshape(-1, 3)
        dummies = Atoms(
            symbols=[dummy_symbol] * len(self.cavities),
            positions=centers,
            cell=self.atoms.get_cell(),
            pbc=self.atoms.get_pbc(),
        )
        combined = self.atoms + dummies
        write(filename, combined, format=format)
