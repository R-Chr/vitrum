import numpy as np
from vitrum.glass_atoms import GlassAtoms
from scipy.stats import linregress
from typing import List, Union, Optional, Tuple
from ase import Atoms


class Diffusion:
    """
    Class for analyzing diffusion in glass structures.
    """
    def __init__(self, trajectory: List[Atoms], sample_times: List[float]):
        """
        Initializes a new instance of the class with the given a trajectory as a list of Atoms objects.

        Args:
            trajectory (List[Atoms]): A list of Atoms objects representing the trajectory.
            sample_times (List[float]): A list of sampled times.
        """

        self.trajectory = [GlassAtoms(atom) for atom in trajectory]
        self.chemical_symbols = np.array(trajectory[0].get_chemical_symbols())
        self.species = np.unique(self.chemical_symbols)
        self.sample_times = sample_times

    def get_mean_square_displacements(self) -> np.ndarray:
        """
        Calculates the mean square displacement for each atom in the trajectory.

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

    def get_diffusion_coef(self, skip_first: int = 100, msds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the diffusion coefficients.

        Args:
            skip_first (int, optional): Number of initial time steps to skip for linear regression. Defaults to 100.
            msds (Optional[np.ndarray], optional): Pre-calculated MSDs. If None, they are calculated.

        Returns:
            np.ndarray: Array of diffusion coefficients.
        """
        if msds is None:
            msds = self.get_mean_square_displacements()
        D = []
        for msd in msds:
            lin_reg = linregress(self.sample_times[skip_first:], msd[skip_first:])
            D.append((lin_reg.slope / 6))
        return np.array(D)

    def get_van_hove_self_correlation(
        self,
        target_atom: str,
        t_window: Optional[int] = None,
        nbin: int = 70
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Van Hove self-correlation function.

        Args:
            target_atom (str): The chemical symbol of the target atom.
            t_window (Optional[int], optional): Time window stride. Defaults to None.
            nbin (int, optional): Number of bins for histogram. Defaults to 70.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - edges: Bin edges (distance).
                - hist: Histogram values (probability).
        """
        index = np.where(self.chemical_symbols == target_atom)[0]

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
            cell = np.diagonal(self.trajectory[start].get_cell())[0]
            dif_pos = current_positions - start_postions
            distances = np.sqrt(np.sum(dif_pos**2, axis=1))
            hist, edges = np.histogram(distances, bins=10 ** np.linspace(np.log10(0.1), np.log10(100), nbin))
            hist_all.append(hist)
        hist = np.mean(np.array(hist_all), axis=0)
        return edges[:-1], hist / len(self.chemical_symbols)

    def get_van_hove_dist_correlation(self):
        pass

    def get_velocity_autocorrelation(self):
        pass
