import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy.interpolate import interpn


def get_dimer_radial_energy(
    calc: Calculator, formula: str, cutoff: float = 8, num_data_points: int = 100
) -> tuple[np.ndarray, list[float]]:
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


def get_pred_energy_forces(atoms: list[Atoms], calc: Calculator) -> tuple[list[float], np.ndarray]:
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
    return pred_energy, np.vstack(pred_forces).flatten()


def eval_plot(reference_data: np.ndarray, predicted_data: np.ndarray, ax: Axes | None = None) -> PathCollection:
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
    reference_data = np.array(reference_data)
    predicted_data = np.array(predicted_data)
    min_max = [
        min(reference_data.min(), predicted_data.min()),
        max(reference_data.max(), predicted_data.max()),
    ]
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
    rmse = float(np.sqrt(np.mean((reference_data - predicted_data) ** 2)))
    ax.set_title(f"RMSE: {rmse * 1000:.2f} meV/atom", loc="left", x=0.05, y=0.90, fontsize=9)
    return ep
