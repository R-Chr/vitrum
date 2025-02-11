import numpy as np
from vitrum.glass_Atoms import glass_Atoms
from scipy.stats import linregress


class diffusion:
    def __init__(self, trajectory: list, sample_times: list):
        """
        Initializes a new instance of the class with the given a trajectory as a list of Atoms objects.

        Parameters:
            trajectory (list): A list of Atoms objects representing the trajectory.
            sample_times (list): A list of sampled times.

        Returns:
            None
        """

        self.trajectory = [glass_Atoms(atom) for atom in trajectory]
        self.chemical_symbols = np.array(trajectory[0].get_chemical_symbols())
        self.species = np.unique(self.chemical_symbols)
        self.sample_times = sample_times

    def get_mean_square_displacements(self):
        """
        Calculates the mean square displacement for each atom in the trajectory.

        Returns:

            ndarray: An array of mean square displacements for each atom in the trajectory.
                for each atom at the corresponding time step.
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

    def get_diffusion_coef(self, skip_first=100, msds=None):
        if msds is None:
            msds = self.get_mean_square_displacements()
        D = []
        for msd in msds:
            lin_reg = linregress(self.sample_times[skip_first:], msd[skip_first:])
            D.append((lin_reg.slope / 6))
        return np.array(D)

    def get_van_hove_self_correlation(self, target_atom, t_window=None, nbin=70):
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
