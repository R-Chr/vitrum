from typing import Dict, List, Optional, Tuple, Union

import diode
import dionysus
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import covalent_radii
from ase.symbols import symbols2numbers
from sklearn.cluster import Birch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


class PersistenceDiagram:
    """
    Persistence diagrams of an atomistic structure, computed via a weighted
    (power) alpha-shape filtration.
    """

    def __init__(
        self,
        atoms: Atoms,
        weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
        weight_scaling: float = 1.0,
        exclude_atoms: Optional[List[str]] = None,
    ):
        """
        Initialize the PersistenceDiagram class.

        Args:
            atoms (Atoms): The structure to analyze.
            weights (Optional[Union[Dict[str, float], List[float], np.ndarray]]):
                Per-atom weighting radius (Angstrom), squared internally into the
                alpha-shape's power weight. Either a dict mapping chemical symbol
                to radius, or an array/list with one radius per atom (indexed
                against `atoms` before `exclude_atoms` filtering, see below).
                Defaults to None, which uses `ase.data.covalent_radii`.
            weight_scaling (float): Factor multiplying every atom's weighting
                radius before it is squared into the alpha-shape weight. Defaults
                to 1.0.
            exclude_atoms (Optional[List[str]]): Chemical symbols to drop from
                `atoms` before computing anything, e.g. to exclude network
                modifiers (which would otherwise fill in loops/voids) from the
                homology analysis. A per-atom `weights` array/list is filtered
                the same way, so it can still be indexed against the original,
                unfiltered `atoms`. Defaults to None (no exclusion).
        """
        if exclude_atoms:
            keep_mask = ~np.isin(np.array(atoms.get_chemical_symbols()), exclude_atoms)
            atoms = atoms[keep_mask]
            if weights is not None and not isinstance(weights, dict):
                weights = np.asarray(weights)[keep_mask]

        self.atoms = atoms
        self.weights = weights
        self.weight_scaling = weight_scaling
        self.diagrams: Optional[Dict[int, pd.DataFrame]] = None
        self._filtration = None
        self._persistence = None
        self._birth_indices: Optional[Dict[int, List[int]]] = None

    def _get_radii(self) -> np.ndarray:
        """
        Resolve `self.weights` and `self.weight_scaling` into a per-atom radius array.
        """
        symbols = np.array(self.atoms.get_chemical_symbols())
        if self.weights is None:
            radii = covalent_radii[symbols2numbers(symbols)].astype(float)
        elif isinstance(self.weights, dict):
            radii = np.array([self.weights[s] for s in symbols], dtype=float)
        else:
            radii = np.asarray(self.weights, dtype=float)
            if radii.shape != (len(self.atoms),):
                raise ValueError(
                    f"`weights` list/array must have one entry per atom ({len(self.atoms)}), got shape {radii.shape}."
                )
        return radii * self.weight_scaling

    def calculate(self, dimensions: Tuple[int, ...] = (1, 2)) -> Dict[int, pd.DataFrame]:
        """
        Calculate the persistence diagram(s) of the structure.

        A single weighted alpha-shape filtration is built (the dominant cost),
        and persistence is read off simultaneously for every requested
        dimension, so asking for both dimensions 1 and 2 costs virtually the
        same as computing just one.

        Args:
            dimensions (Tuple[int, ...]): Homology dimensions to compute
                persistence diagrams for. 1 corresponds to loops/rings, 2 to
                voids/cavities. Defaults to (1, 2).

        Returns:
            Dict[int, pandas.DataFrame]: One "Birth"/"Death" DataFrame per
                requested dimension.
        """
        coords = self.atoms.get_positions()
        radii = self._get_radii()
        points = np.column_stack([coords, radii**2])

        simplices = diode.fill_weighted_alpha_shapes(points)
        f = dionysus.Filtration(simplices)
        m = dionysus.homology_persistence(f)
        dgms = dionysus.init_diagrams(m, f)

        # Kept around so `get_cycle_atoms`/`get_diagram_composition` can later resolve
        # each point's representative cycle from the persistence pairing, without
        # having to recompute or re-store the filtration themselves.
        self._filtration = f
        self._persistence = m
        self._birth_indices = {dim: [p.data for p in dgms[dim]] for dim in dimensions}

        self.diagrams = {
            dim: pd.DataFrame(
                {
                    "Birth": [p.birth for p in dgms[dim]],
                    "Death": [p.death for p in dgms[dim]],
                }
            )
            for dim in dimensions
        }
        return self.diagrams

    def get_diagram(self, dimension: int = 1) -> pd.DataFrame:
        """
        Get the persistence diagram for a given homology dimension.

        Args:
            dimension (int): Homology dimension requested. Must have been
                included in the `dimensions` passed to `calculate()`.

        Returns:
            pandas.DataFrame: Birth/Death pairs for this dimension.

        Raises:
            ValueError: If `calculate()` has not been run yet, or `dimension`
                was not one of the dimensions it was run with.
        """
        if self.diagrams is None:
            raise ValueError("Persistence diagrams have not been calculated yet. Run .calculate() first.")
        if dimension not in self.diagrams:
            raise ValueError(
                f"Dimension {dimension} was not calculated; available dimensions: {sorted(self.diagrams)}. "
                "Re-run .calculate() with it included in `dimensions`."
            )
        return self.diagrams[dimension]

    def _get_cycle_atoms(self, birth_index: int) -> Optional[np.ndarray]:
        """
        Resolve the atom indices making up the representative cycle of the
        homology class born at filtration index `birth_index`.

        Every finite-persistence point is paired by `calculate()`'s reduction
        with a "death" simplex whose reduced boundary chain is exactly the
        cycle that collapses at that point (`m.pair(birth_index)` gives that
        simplex's filtration index directly, sidestepping any birth/death
        *value* matching, which is unreliable whenever multiple simplices
        share a filtration value). The chain's entries reference simplices
        one dimension lower than the death simplex (edges for loops, triangles
        for voids); the cycle's atoms are the union of all their vertices.

        Returns None for an essential class (no death simplex, i.e. infinite
        persistence).
        """
        death_index = self._persistence.pair(birth_index)
        if death_index == self._persistence.unpaired:
            return None
        atom_indices = set()
        for entry in self._persistence[death_index]:
            atom_indices.update(self._filtration[entry.index])
        return np.array(sorted(atom_indices))

    def get_cycle_atoms(self, dimension: int = 1, index: int = 0) -> np.ndarray:
        """
        Get the atom indices making up a loop's or void's representative cycle.

        Args:
            dimension (int): Homology dimension the loop/void belongs to (1
                for loops, 2 for voids).
            index (int): Row index into `get_diagram(dimension)` identifying
                which loop/void to resolve.

        Returns:
            np.ndarray: Sorted, unique indices into `self.atoms` of the atoms
                making up the representative cycle.

        Raises:
            ValueError: If `calculate()` has not been run yet, `dimension`
                was not one of the dimensions it was run with, `index` is out
                of range, or the point has infinite persistence (no death
                simplex, so no cycle can be resolved).
        """
        dgm = self.get_diagram(dimension)
        if not 0 <= index < len(dgm):
            raise ValueError(f"`index` {index} out of range for dimension {dimension} diagram of length {len(dgm)}.")

        atoms = self._get_cycle_atoms(self._birth_indices[dimension][index])
        if atoms is None:
            raise ValueError(
                f"Point {index} in dimension {dimension} has infinite persistence (Death = inf); "
                "no death simplex exists to recover a representative cycle from."
            )
        return atoms

    def get_diagram_composition(self, dimension: int = 1) -> pd.DataFrame:
        """
        Get the persistence diagram for a dimension, annotated with the
        atomic composition of each loop's or void's representative cycle.

        Building on `get_diagram`, this adds:

        - "Lifetime": Death - Birth.
        - "Mean_age": (Birth + Death) / 2 (see `get_apf`).
        - "Atoms": the atom indices making up the representative cycle (see
          `get_cycle_atoms`).
        - "Size": the number of atoms in the cycle.
        - "N_<symbol>": the count of each chemical species present in
          `self.atoms` within the cycle, one column per species.

        Points with infinite persistence have no death simplex to recover a
        cycle from, so their "Atoms"/"Size"/"N_<symbol>" entries are left as
        None/NaN.

        Args:
            dimension (int): Homology dimension of the diagram to annotate.

        Returns:
            pandas.DataFrame: `get_diagram(dimension)` with the columns above
                appended.
        """
        dgm = self.get_diagram(dimension).copy()
        symbols = np.array(self.atoms.get_chemical_symbols())
        species = np.unique(symbols)

        cycles = [self._get_cycle_atoms(birth_index) for birth_index in self._birth_indices[dimension]]

        dgm["Lifetime"] = dgm["Death"] - dgm["Birth"]
        dgm["Mean_age"] = (dgm["Birth"] + dgm["Death"]) / 2
        dgm["Atoms"] = cycles
        dgm["Size"] = [len(c) if c is not None else np.nan for c in cycles]
        for s in species:
            dgm[f"N_{s}"] = [float(np.count_nonzero(symbols[c] == s)) if c is not None else np.nan for c in cycles]
        return dgm

    def get_apf(self, dimension: int = 1) -> pd.DataFrame:
        """
        Calculate the accumulated persistence function (APF) of a persistence diagram.

        Following Biscio & Møller (J. Comput. Graph. Stat. 2019, 28, 671) and its
        application to glass structure by Sørensen et al. (Sci. Adv. 2020, 6,
        eabc2320), each birth-death pair (b, d) is assigned a "mean age"
        m = (b + d) / 2; APF(t) = sum of (d - b) over all pairs with m <= t.
        Sorting pairs by mean age and cumulatively summing their persistence
        (d - b) collapses the whole diagram into a single monotonic curve,
        which is easier to compare across structures than a scatter of points.

        Infinite-persistence points (Death == inf, e.g. the single essential
        dimension-0 class) are dropped, since their persistence is undefined.

        Args:
            dimension (int): Homology dimension of the diagram to summarize.

        Returns:
            pandas.DataFrame: Columns "Mean_age" ((Birth + Death) / 2, ascending)
                and "APF" (cumulative sum of Death - Birth in that order).

        """
        dgm = self.get_diagram(dimension)
        finite = dgm[np.isfinite(dgm["Death"])]
        mean_age = ((finite["Birth"] + finite["Death"]) / 2).to_numpy()
        order = np.argsort(mean_age)
        lifetimes = (finite["Death"] - finite["Birth"]).to_numpy()[order]
        return pd.DataFrame({"Mean_age": mean_age[order], "APF": np.cumsum(lifetimes)})

    def get_sph(
        self,
        dimension: int = 1,
        q_values: Optional[np.ndarray] = None,
        reference_radius: Optional[float] = None,
        reference_symbol: str = "O",
        sigma: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Calculate the S_PH(Q) function of a persistence diagram.

        Following Sørensen et al. (Sci. Adv. 2020, 6, eabc2320, Eq. 2), each
        birth-death pair (b, d) is converted to a reciprocal-space contribution
        at Q = 2*pi / l(d), where l(d) = 2 * sqrt(d + r**2) is the diameter, at
        the loop's death time d, of a probe atom of radius r.

        S_PH(Q) is formally a sum of Dirac delta functions, with each contribution
        broadened here into a Gaussian of width `sigma`.

        Args:
            dimension (int): Homology dimension of the diagram to summarize.
            q_values (Optional[np.ndarray]): Q values (Angstrom^-1) to
                evaluate S_PH(Q) on. 
            reference_radius (Optional[float]): Radius (Angstrom) of the
                reference atom used in the diameter conversion l(d). Defaults
                to None, which resolves `reference_symbol`'s radius from
                `self.weights` (falling back to `ase.data.covalent_radii` if
                `self.weights` is None), scaled by `self.weight_scaling` for
                consistency with how `calculate()` weighted the atoms.
            reference_symbol (str): Chemical symbol used to resolve
                `reference_radius` when not given explicitly. Defaults to
                "O", following the reference paper's choice for oxide
                glasses (oxygen has the largest radius and is the most
                abundant element in the studied loops).
            sigma (Optional[float]): Standard deviation (Angstrom^-1) of the
                Gaussian broadening applied to each delta-function
                contribution. Defaults to None, which uses 2% of the sampled
                Q range.

        Returns:
            pandas.DataFrame: Columns "Q" and "S_PH".

        Raises:
            ValueError: If `calculate()` has not been run yet, `dimension`
                was not one of the dimensions it was run with, or
                `reference_radius` could not be resolved from `self.weights`.
        """
        dgm = self.get_diagram(dimension)
        finite = dgm[np.isfinite(dgm["Death"])]
        deaths = finite["Death"].to_numpy()

        if reference_radius is None:
            if isinstance(self.weights, dict):
                if reference_symbol not in self.weights:
                    raise ValueError(
                        f"`reference_symbol` '{reference_symbol}' not found in `self.weights`; "
                        "pass `reference_radius` explicitly instead."
                    )
                reference_radius = self.weights[reference_symbol] * self.weight_scaling
            elif self.weights is None:
                reference_radius = (
                    float(covalent_radii[symbols2numbers([reference_symbol])[0]]) * self.weight_scaling
                )
            else:
                raise ValueError(
                    "`self.weights` is a per-atom list/array, which does not identify a radius "
                    "for `reference_symbol`; pass `reference_radius` explicitly instead."
                )

        q_peaks = 2 * np.pi / (2 * np.sqrt(deaths + reference_radius**2)) if len(deaths) else np.array([])

        if q_values is None:
            if len(q_peaks):
                q_values = np.linspace(q_peaks.min() * 0.5, q_peaks.max() * 1.5, 200)
            else:
                q_values = np.linspace(0.0, 10.0, 200)
        else:
            q_values = np.asarray(q_values, dtype=float)
        if sigma is None:
            sigma = 0.02 * (q_values.max() - q_values.min()) or 0.05

        if len(q_peaks) == 0:
            s_ph = np.zeros_like(q_values)
        else:
            diffs = q_values[:, None] - q_peaks[None, :]
            gaussians = np.exp(-(diffs**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
            s_ph = gaussians.sum(axis=1) / len(q_peaks)

        return pd.DataFrame({"Q": q_values, "S_PH": s_ph})

    def get_persistence_image(
        self,
        dimension: int = 1,
        resolution: int = 20,
        sigma: Optional[float] = None,
        birth_range: Optional[Tuple[float, float]] = None,
        persistence_range: Optional[Tuple[float, float]] = None,
        weight_fn: Optional[callable] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Calculate the persistence image of a persistence diagram.

        Following Adams et al. (J. Mach. Learn. Res. 2017, 18, 1), each
        birth-death pair (b, d) is first mapped to birth-persistence
        coordinates (b, p) with p = d - b, then represented as a Gaussian of
        width `sigma` centered at (b, p) and scaled by `weight_fn(p)`.

        Infinite-persistence points are dropped, as for `get_apf`.

        Args:
            dimension (int): Homology dimension of the diagram to summarize.
            resolution (int): Number of grid points along each axis; the
                returned image has shape (resolution, resolution).
            sigma (Optional[float]): Standard deviation of the Gaussian placed
                at each point.
            birth_range (Optional[Tuple[float, float]]): (min, max) birth
                value range to sample over.
            persistence_range (Optional[Tuple[float, float]]): (min, max)
                persistence (Death - Birth) value range to sample over.
            weight_fn (Optional[callable]): Function mapping an array of
                persistence values to weights. 

        Returns:
            Tuple[np.ndarray, Tuple[float, float, float, float]]: (image,
                extent). `image` has shape (resolution, resolution), rows
                indexed by persistence and columns by birth. `extent`
                is (birth_min, birth_max, persistence_min, persistence_max).

        Raises:
            ValueError: If `calculate()` has not been run yet, or `dimension`
                was not one of the dimensions it was run with.
        """
        dgm = self.get_diagram(dimension)
        finite = dgm[np.isfinite(dgm["Death"])]
        births = finite["Birth"].to_numpy()
        persistences = (finite["Death"] - finite["Birth"]).to_numpy()

        if birth_range is None:
            if len(births):
                b_min, b_max = float(births.min()), float(births.max())
                pad = (b_max - b_min) * 0.1 or 1.0
                birth_range = (b_min - pad, b_max + pad)
            else:
                birth_range = (0.0, 1.0)
        if persistence_range is None:
            p_max = float(persistences.max()) if len(persistences) else 1.0
            persistence_range = (0.0, p_max * 1.1 or 1.0)
        if sigma is None:
            sigma = 0.1 * (persistence_range[1] - persistence_range[0]) or 1.0
        if weight_fn is None:
            weight_fn = lambda p: p  # noqa: E731

        xs = np.linspace(*birth_range, resolution)
        ys = np.linspace(*persistence_range, resolution)
        extent = (birth_range[0], birth_range[1], persistence_range[0], persistence_range[1])

        if len(births) == 0:
            return np.zeros((resolution, resolution)), extent

        weights = weight_fn(persistences)
        dx = xs[None, :, None] - births[None, None, :]
        dy = ys[:, None, None] - persistences[None, None, :]
        gaussians = np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        image = (gaussians * weights[None, None, :]).sum(axis=2)
        return image, extent

    def plot_diagram(self, dimension: int = 1, ax=None, **plot_kwargs):
        """
        Plot the persistence diagram (birth vs. death scatter) using matplotlib.

        Args:
            dimension (int): Homology dimension to plot.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure and axes are created.
            **plot_kwargs: Additional keyword arguments passed to `ax.scatter`.

        Returns:
            matplotlib.axes.Axes: The axes the diagram was plotted on.
        """
        import matplotlib.pyplot as plt

        dgm = self.get_diagram(dimension)
        finite = dgm[np.isfinite(dgm["Death"])]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
            top = float(max(finite["Death"].max(), finite["Birth"].max()) * 1.05) if len(finite) else 1.0
            ax.plot([0, top], [0, top], "k--", linewidth=1)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_xlabel("Birth (Å$^2$)", fontsize=12)
            ax.set_ylabel("Death (Å$^2$)", fontsize=12)
            ax.set_title(f"H$_{dimension}$ persistence diagram")

        ax.scatter(finite["Birth"], finite["Death"], **plot_kwargs)
        return ax

    def plot_apf(self, dimension: int = 1, ax=None, **plot_kwargs):
        """
        Plot the accumulated persistence function (APF) using matplotlib.

        Args:
            dimension (int): Homology dimension to plot.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure and axes are created.
            **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

        Returns:
            matplotlib.axes.Axes: The axes the APF was plotted on.
        """
        import matplotlib.pyplot as plt

        apf = self.get_apf(dimension)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
            ax.set_xlabel("Mean age, (Birth + Death) / 2 (Å$^2$)", fontsize=12)
            ax.set_ylabel("APF (Å$^2$)", fontsize=12)
            ax.set_title(f"H$_{dimension}$ accumulated persistence function")

        ax.plot(apf["Mean_age"], apf["APF"], **plot_kwargs)
        return ax

    def plot_sph(
        self,
        dimension: int = 1,
        q_values: Optional[np.ndarray] = None,
        reference_radius: Optional[float] = None,
        reference_symbol: str = "O",
        sigma: Optional[float] = None,
        ax=None,
        **plot_kwargs,
    ):
        """
        Plot the S_PH(Q) function using matplotlib.

        Args:
            dimension (int): Homology dimension to plot.
            q_values (Optional[np.ndarray]): Q values to evaluate over.
                Defaults to None (see `get_sph`).
            reference_radius (Optional[float]): Reference atom radius.
                Defaults to None (see `get_sph`).
            reference_symbol (str): Reference atom chemical symbol. Defaults
                to "O".
            sigma (Optional[float]): Gaussian broadening width. Defaults to
                None (see `get_sph`).
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure and axes are created.
            **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

        Returns:
            matplotlib.axes.Axes: The axes S_PH(Q) was plotted on.
        """
        import matplotlib.pyplot as plt

        sph = self.get_sph(
            dimension,
            q_values=q_values,
            reference_radius=reference_radius,
            reference_symbol=reference_symbol,
            sigma=sigma,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
            ax.set_xlabel("Q (Å$^{-1}$)", fontsize=12)
            ax.set_ylabel("S$_{PH}$(Q)", fontsize=12)
            ax.set_title(f"H$_{dimension}$ S$_{{PH}}$(Q)")

        ax.plot(sph["Q"], sph["S_PH"], **plot_kwargs)
        return ax

    def plot_persistence_image(
        self,
        dimension: int = 1,
        resolution: int = 20,
        sigma: Optional[float] = None,
        birth_range: Optional[Tuple[float, float]] = None,
        persistence_range: Optional[Tuple[float, float]] = None,
        weight_fn: Optional[callable] = None,
        ax=None,
        **imshow_kwargs,
    ):
        """
        Plot the persistence image using matplotlib.

        Args:
            dimension (int): Homology dimension to plot.
            resolution (int): Number of grid points along each axis.
            sigma (Optional[float]): Standard deviation of the Gaussian placed
                at each point. Defaults to None (see `get_persistence_image`).
            birth_range (Optional[Tuple[float, float]]): (min, max) birth
                value range to sample over. Defaults to None (see
                `get_persistence_image`).
            persistence_range (Optional[Tuple[float, float]]): (min, max)
                persistence value range to sample over. Defaults to None (see
                `get_persistence_image`).
            weight_fn (Optional[callable]): Function mapping an array of
                persistence values to weights. Defaults to None (see
                `get_persistence_image`).
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure and axes are created.
            **imshow_kwargs: Additional keyword arguments passed to `ax.imshow`.

        Returns:
            matplotlib.axes.Axes: The axes the image was plotted on.
        """
        import matplotlib.pyplot as plt

        image, extent = self.get_persistence_image(
            dimension,
            resolution=resolution,
            sigma=sigma,
            birth_range=birth_range,
            persistence_range=persistence_range,
            weight_fn=weight_fn,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))
            ax.set_xlabel("Birth (Å$^2$)", fontsize=12)
            ax.set_ylabel("Persistence, Death - Birth (Å$^2$)", fontsize=12)
            ax.set_title(f"H$_{dimension}$ persistence image")

        im = ax.imshow(image, extent=extent, origin="lower", aspect="auto", **imshow_kwargs)
        plt.colorbar(im, ax=ax)
        return ax


"""
NOTE: `LocalPD` and the standalone `get_local_persistence` below are currently
broken and quarantined (raise NotImplementedError on use).
See docs/vitrum/known_issues.md.
"""

class LocalPD:  # Broken after moving persistence diagram functions out of glass_Atoms
    def __init__(
        self,
        glass_atoms_list,
        center_atom,
        cutoff,
        dimension=1,
        weights=None,
        birch_threshold=0.075,
    ):
        raise NotImplementedError(
            "LocalPD is currently broken: it calls neighborhood.get_persistence_diagram(...) "
            "as a bound method, but that functionality now lives in the standalone "
            "PersistenceDiagram class in this module and this call site was never updated. "
            "See docs/vitrum/known_issues.md."
        )
        self.atom_list = glass_atoms_list
        self.center_atom = center_atom
        self.cutoff = cutoff
        self.dimension = dimension
        self.weights = weights
        self.birch_threshold = birch_threshold

    def compute_features(self):
        sampling_centers = self.find_sampling_centers()
        features = []
        for atoms in self.atom_list:
            peristence_diagrams = self.get_local_persistence(atoms, self.center_atom, self.cutoff)
            features.append(self.kde_pd(sampling_centers, peristence_diagrams))
        return np.vstack(features)

    def center_atoms(self, atoms, center_atom):
        dim = np.diagonal(atoms.get_cell())
        positions = atoms.get_positions()
        x_dif = positions[:, 0] - positions[center_atom, 0]
        y_dif = positions[:, 1] - positions[center_atom, 1]
        z_dif = positions[:, 2] - positions[center_atom, 2]
        x_dif = np.where(
            x_dif > 0.5 * dim[0],
            positions[:, 0] - positions[center_atom, 0] - dim[0],
            x_dif,
        )
        y_dif = np.where(
            y_dif > 0.5 * dim[1],
            positions[:, 1] - positions[center_atom, 1] - dim[1],
            y_dif,
        )
        z_dif = np.where(
            z_dif > 0.5 * dim[2],
            positions[:, 2] - positions[center_atom, 2] - dim[2],
            z_dif,
        )
        x_dif = np.where(
            x_dif < -0.5 * dim[0],
            positions[:, 0] - positions[center_atom, 0] + dim[0],
            x_dif,
        )
        y_dif = np.where(
            y_dif < -0.5 * dim[1],
            positions[:, 1] - positions[center_atom, 1] + dim[1],
            y_dif,
        )
        z_dif = np.where(
            z_dif < -0.5 * dim[2],
            positions[:, 2] - positions[center_atom, 2] + dim[2],
            z_dif,
        )
        new_postions = np.vstack([x_dif, y_dif, z_dif]).T
        return new_postions

    def get_local_persistence(self, atom, center_id, cutoff):
        persistence_diagrams = []
        if isinstance(center_id, str):
            types = atom.get_chemical_symbols()
        if isinstance(center_id, int):
            types = atom.get_atomic_numbers()
        centers = np.where(np.array(types) == center_id)[0]
        for i in tqdm(centers):
            neighbors = np.where(atom.get_dist()[i, :] < cutoff)[0]
            neighborhood = atom[neighbors]
            center_index = np.where(neighbors == i)
            neighborhood.set_positions(self.center_atoms(neighborhood, center_index))
            persistence_diagrams.append(
                neighborhood.get_persistence_diagram(dimension=self.dimension, weights=self.weights)
            )
        return persistence_diagrams

    def find_sampling_centers(self):
        peristence_diagrams = self.get_local_persistence(self.atom_list[0], self.center_atom, self.cutoff)
        total_df = pd.concat(peristence_diagrams)
        birth_death = np.array([total_df["Birth"], total_df["Death"] - total_df["Birth"]]).T
        birch = Birch(n_clusters=100, threshold=self.birch_threshold).fit(birth_death)
        return birch.subcluster_centers_

    def kde_pd(self, centers, list_pds):
        features = []
        for pds in list_pds:
            data = np.vstack((pds["Birth"], pds["Death"] - pds["Birth"])).T
            kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(data)
            features.append(np.exp(kde.score_samples(centers)))
        features = np.array(features)
        return features

def get_local_persistence(atoms, center_id, cutoff):
    """
    Calculate the persistence diagram of the local environment of an atom.

    Parameters:
        center_id (int or str): The atomic number or symbol of the central atom.
        cutoff (float): The cutoff distance for the local environment.

    Returns:
        list: A list of pandas.DataFrame containing the persistence diagram of the local environment.
    """
    raise NotImplementedError(
        "get_local_persistence is currently broken: it calls neighborhood.center() "
        "(which does not do minimum-image centering on an ASE Atoms object) and "
        "neighborhood.get_persistence_diagram() (which does not exist as a bound method; "
        "use PersistenceDiagram(neighborhood).calculate(...) instead). "
        "See docs/vitrum/known_issues.md."
    )
    persistence_diagrams = []
    if isinstance(center_id, str):
        types = atoms.get_chemical_symbols()
    if isinstance(center_id, int):
        types = atoms.get_atomic_numbers()
    centers = np.where(types == center_id)[0]
    for i in centers:
        neighbors = np.where(atoms.get_dist()[i, :] < cutoff)[0]
        neighborhood = atoms[neighbors]
        neighborhood.center()
        persistence_diagrams.append(neighborhood.get_persistence_diagram())
    return persistence_diagrams
