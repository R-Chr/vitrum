import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import Birch
from sklearn.neighbors import KernelDensity
import dionysus
import diode


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


def get_persistence_diagram(atoms, dimension=1, weights=None):
    """
    Calculate the persistence diagram of the given data points.

    Parameters:
        dimension (int, optional): The dimension of the persistence diagram to calculate. Defaults to 1.
        weights (dict or list, optional): The weights to assign to each data point. Can be a dictionary mapping chemical symbols to weights or a list of weights. Defaults to None.

    Returns:
        pandas.DataFrame: The persistence diagram as a DataFrame with columns "Birth" and "Death".
    """
    coord = atoms.get_positions()
    data = np.column_stack([atoms.get_chemical_symbols(), coord])
    dfpoints = pd.DataFrame(data, columns=["Atom", "x", "y", "z"])
    chem_species = np.unique(atoms.get_chemical_symbols())

    if weights is None:
        radii = [0 for i in chem_species]
    elif isinstance(weights, dict):
        radii = [weights[i] for i in chem_species]
    elif isinstance(weights, list):
        radii = weights

    conditions = [(dfpoints["Atom"] == i) for i in chem_species]
    choice_weight = [i**2 for i in radii]

    dfpoints["w"] = np.select(conditions, choice_weight)
    dfpoints["x"] = pd.to_numeric(dfpoints["x"])
    dfpoints["y"] = pd.to_numeric(dfpoints["y"])
    dfpoints["z"] = pd.to_numeric(dfpoints["z"])

    points = dfpoints[["x", "y", "z", "w"]].to_numpy()
    simplices = diode.fill_weighted_alpha_shapes(points)
    f = dionysus.Filtration(simplices)
    m = dionysus.homology_persistence(f)
    dgms = dionysus.init_diagrams(m, f)

    # Gather the PD of loop in a dataframe
    dfPD = pd.DataFrame(
        data={
            "Birth": [p.birth for p in dgms[dimension]],
            "Death": [p.death for p in dgms[dimension]],
        }
    )
    return dfPD


def get_local_persistence(atoms, center_id, cutoff):
    """
    Calculate the persistence diagram of the local environment of an atom.

    Parameters:
        center_id (int or str): The atomic number or symbol of the central atom.
        cutoff (float): The cutoff distance for the local environment.

    Returns:
        list: A list of pandas.DataFrame containing the persistence diagram of the local environment.
    """
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
