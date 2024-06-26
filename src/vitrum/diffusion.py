import numpy as np


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

        self.trajectory = trajectory
        self.chemical_symbols = trajectory[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.sample_times = sample_times

    def calculate_mean_square_displacement(self):
        """
        Calculates the mean square displacement for each atom in the trajectory.

        Returns:
            list: A list of NumPy arrays, where each array represents the mean square displacement
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
            indices = np.where(np.array(self.chemical_symbols) == species)[0]
            mean_square_displacement.append(np.mean(displacement_array[:, indices], axis=1))

        return mean_square_displacement

    def calculate_diffusion_coefficients(self):
        pass

    def calculate_van_hove_self_correlation(self):
        pass
