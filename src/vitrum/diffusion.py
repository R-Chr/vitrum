import numpy as np
from ase import Atoms
from scipy.stats import linregress

from vitrum.trajectory_tools import unwrap_trajectory

# How far outside the cell, as a fraction of a cell length, a "wrapped" trajectory may sit.
# MD codes write wrapped coordinates that land past a face: the example trajectory in
# examples/analysis reaches 0.027 of a cell outside. What this guard is for is a trajectory
# that was already unwrapped, where atoms leave by whole cells, so the two stay far apart.
# ponytail: a fixed fraction, not a per-trajectory judgement; a short unwrapped trajectory
# whose atoms never travel a tenth of a cell is accepted, and unwrapping it again is a no-op.
_WRAP_TOLERANCE = 0.1


class Diffusion:
    """
    Class for analyzing diffusion in glass structures.
    """

    def __init__(self, trajectory: list[Atoms], sample_times: list[float], wrapped: bool = True):
        """
        Initializes a new instance of the class with the given a trajectory as a list of Atoms objects.

        Args:
            trajectory (List[Atoms]): A list of Atoms objects representing the trajectory.
            sample_times (List[float]): A list of sampled times.
            wrapped (bool, optional): Whether `trajectory` positions are PBC-wrapped and need
                unwrapping before computing displacements. Set to False if you've already
                unwrapped the trajectory yourself (e.g. via `vitrum.trajectory.unwrap_trajectory`).
                Defaults to True.
        """

        if wrapped:
            for atoms in trajectory:
                scaled_positions = atoms.get_scaled_positions(wrap=False)
                overhang = max(-scaled_positions.min(), scaled_positions.max() - 1.0, 0.0)
                if overhang > _WRAP_TOLERANCE:
                    raise ValueError(
                        f"wrapped=True but some atom coordinates lie {overhang:.2f} of a cell "
                        "length outside it. Set wrapped=False if the trajectory is already "
                        "unwrapped."
                    )
            trajectory = unwrap_trajectory(trajectory)
        if len(sample_times) != len(trajectory):
            raise ValueError(
                f"sample_times has {len(sample_times)} entries but the trajectory has "
                f"{len(trajectory)} frames; they must correspond one-to-one."
            )
        self.trajectory = list(trajectory)
        self.chemical_symbols = np.array(trajectory[0].get_chemical_symbols())
        self.species = np.unique(self.chemical_symbols)
        self.sample_times = sample_times

    def get_mean_square_displacements(self) -> np.ndarray:
        """
        Calculates the mean square displacement for each atom in the trajectory.

        Note:
            Displacements are measured from a **single time origin** (the first frame) rather
            than averaged over multiple origins. This is noisier than the standard windowed
            multi-origin estimator, particularly at long lag times where few independent
            samples remain. See `docs/vitrum/known_issues.md`.

        Returns:
            np.ndarray: An array of mean square displacements.
                Rows correspond to:
                0: Total MSD
                1+: MSD for each species in self.species order.
        """
        initial_positions = self.trajectory[0].get_positions()
        displacement_array = np.zeros((len(self.trajectory), len(self.chemical_symbols)))

        for time_step, atoms in enumerate(self.trajectory):
            positions = atoms.get_positions()
            displacements = positions - initial_positions
            displacement_array[time_step, :] = np.sum(displacements**2, axis=1)

        mean_square_displacement = []
        mean_square_displacement.append(np.mean(displacement_array, axis=1))

        for species in self.species:
            indices = np.where(self.chemical_symbols == species)[0]
            mean_square_displacement.append(np.mean(displacement_array[:, indices], axis=1))

        return np.array(mean_square_displacement)

    def get_diffusion_coef(self, skip_first: int = 100, msds: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate the diffusion coefficients.

        Args:
            skip_first (int, optional): Number of initial time steps to skip for linear regression. Defaults to 100.
            msds (Optional[np.ndarray], optional): Pre-calculated MSDs. If None, they are calculated.

        Returns:
            np.ndarray: Array of diffusion coefficients.

        Raises:
            ValueError: If `skip_first` leaves fewer than 3 frames to fit.
        """
        n_frames = len(self.sample_times)
        if n_frames - skip_first < 3:
            raise ValueError(
                f"skip_first={skip_first} leaves {max(n_frames - skip_first, 0)} of {n_frames} "
                "frames; need at least 3 points for a linear fit. Lower skip_first or use a "
                "longer trajectory."
            )
        if msds is None:
            msds = self.get_mean_square_displacements()
        D = []
        for msd in msds:
            lin_reg = linregress(self.sample_times[skip_first:], msd[skip_first:])
            D.append(lin_reg.slope / 6)
        return np.array(D)

    def get_van_hove_self_correlation(
        self, target_atom: str, t_window: int | None = None, nbin: int = 70
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Van Hove self-correlation function.

        Args:
            target_atom (str): The chemical symbol of the target atom.
            t_window (Optional[int], optional): Time window stride. Defaults to None.
            nbin (int, optional): Number of bins for histogram. Defaults to 70.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - edges: Bin edges (distance).
                - hist: Histogram values, normalised per atom of `target_atom`.

        Raises:
            ValueError: If `target_atom` is not present in the trajectory.
        """
        index = np.where(self.chemical_symbols == target_atom)[0]
        if index.size == 0:
            raise ValueError(
                f"target_atom '{target_atom}' not present in the trajectory. Available species: {list(self.species)}."
            )

        if t_window is None:
            start_indicies = [0]
            end_indicies = [-1]
        else:
            start_indicies = np.arange(0, len(self.sample_times) - t_window, t_window)
            end_indicies = np.arange(t_window, len(self.sample_times), t_window)

        hist_all = []
        for start, end in zip(start_indicies, end_indicies):
            start_postions = self.trajectory[start].get_positions()[index]
            current_positions = self.trajectory[end].get_positions()[index]
            dif_pos = current_positions - start_postions
            distances = np.sqrt(np.sum(dif_pos**2, axis=1))
            hist, edges = np.histogram(distances, bins=10 ** np.linspace(np.log10(0.1), np.log10(100), nbin))
            hist_all.append(hist)
        hist = np.mean(np.array(hist_all), axis=0)
        # Only atoms of the target species are histogrammed, so they are what normalises it.
        return edges[:-1], hist / len(index)
