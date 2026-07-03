import numpy as np
import pandas as pd
from ase import Atoms
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error


def get_dimer_radial_energy(calc, formula, cutoff=8, num_data_points=100):
    """
    Calculate the predicted energy of a dimer as a function of separation distance.

    Args:
        calc (ase.calculators.calculator.Calculator): The (MLIP) calculator to evaluate energies with.
        formula (str): The two-atom chemical formula of the dimer, e.g. "OO".
        cutoff (float, optional): Maximum separation distance to sample, in Angstrom. Defaults to 8.
        num_data_points (int, optional): Number of distances to sample between 0.1 and cutoff. Defaults to 100.

    Returns:
        Tuple[np.ndarray, List[float]]: The sampled distances and the corresponding predicted energies.
    """
    pred_energy = []
    distances = np.linspace(0.1, cutoff, num_data_points)
    for d in distances:
        atoms = Atoms(formula, positions=[(0, 0, 0), (0, 0, d)], pbc=False)
        atoms.calc = calc
        pred_energy.append(atoms.get_potential_energy())
    return distances, pred_energy


def get_pred_energy_forces(atoms, calc):
    """
    Calculate predicted per-atom energies and forces for a list of structures.

    Args:
        atoms (List[ase.Atoms]): The structures to evaluate.
        calc (ase.calculators.calculator.Calculator): The (MLIP) calculator to evaluate energies/forces with.

    Returns:
        Tuple[List[float], np.ndarray]: Per-structure energy per atom, and a flattened array of all predicted forces.
    """
    pred_energy = []
    pred_forces = []
    for a in atoms:
        a.calc = calc
        pred_energy.append(a.get_potential_energy() / len(a))
        pred_forces.append(a.get_forces())
    pred_forces = np.vstack(pred_forces).flatten()
    return pred_energy, pred_forces


def min_max_val(array1, array2):
    """
    Find the combined minimum and maximum value across two arrays.

    Args:
        array1 (array-like): The first array.
        array2 (array-like): The second array.

    Returns:
        List[float]: A two-element list, [min_val, max_val].
    """
    max_val = np.max([np.max(array1), np.max(array2)])
    min_val = np.min([np.min(array1), np.min(array2)])
    return [min_val, max_val]


def eval_plot(reference_data, predicted_data, ax=None):
    """
    Plot a density-colored parity plot of predicted vs. reference data, annotated with RMSE.

    Args:
        reference_data (array-like): The reference (e.g. DFT) values.
        predicted_data (array-like): The predicted (e.g. MLIP) values.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to the current axes.

    Returns:
        matplotlib.collections.PathCollection: The scatter plot artist.
    """
    if ax is None:
        ax = plt.gca()
    min_max = min_max_val(reference_data, predicted_data)
    reference_data = np.array(reference_data)
    predicted_data = np.array(predicted_data)
    data, x_e, y_e = np.histogram2d(reference_data, predicted_data, bins=20, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([reference_data, predicted_data]).T,
        method="splinef2d",
        bounds_error=False,
    )
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    reference_data, predicted_data, z = reference_data[idx], predicted_data[idx], z[idx]

    ep = ax.scatter(reference_data, predicted_data, c=z, s=7, rasterized=True)
    ax.plot(min_max, min_max, "-k")
    ax.set_xlim(min_max[0], min_max[1])
    ax.set_ylim(min_max[0], min_max[1])
    rmse = root_mean_squared_error(reference_data, predicted_data)
    ax.set_title(f"RMSE: {rmse*1000:.2f} meV/atom", loc="left", x=0.05, y=0.90, fontsize=9)
    return ep
