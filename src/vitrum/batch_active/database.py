import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def update_ace_database(
    wd: str,
    atoms: list,
    iteration: int,
    force_threshold: int = 100,
    train_test_method: str = "composition",
    train_test_fraction: float = 0.2,
    database_paths=None,
    metadata=None,
):
    """
    Update the ACE database with new structures.

    Parameters:
        wd (str): Working directory.
        atoms (list): List of ASE atoms objects.
        iteration (int): Current iteration number.
        force_threshold (int): Force threshold for filtering structures.
        train_test_method (str): Method for splitting data into train and test sets.
        train_test_split (float): Fraction of data to be used for testing.
        database_paths (dict): Paths to the existing train and test databases.
        metadata (str): Metadata for the structures.
    """
    energy = [i.get_total_energy() for i in atoms]
    force = [i.get_forces().tolist() for i in atoms]
    stress = np.array([i.get_stress() for i in atoms])
    data = {"energy": energy, "forces": force, "stress": stress, "ase_atoms": atoms, "iteration": iteration}
    if metadata:
        data["sample_type"] = metadata
    # create a DataFrame
    df = pd.DataFrame(data)
    print(f"Iteration {iteration} has {len(df)} structures")
    df = df[~df["forces"].apply(lambda x: np.max(x) > force_threshold)]
    df = df[~df["forces"].apply(lambda x: np.min(x) < -force_threshold)]
    print(f"{len(df)} structures remain after force threshold filter")

    if train_test_method == "random":
        # Randomly split the data into train and test sets
        df_new = train_test_split(df, test_size=train_test_fraction, random_state=42)

    elif train_test_method == "composition":
        # determine train/test split
        composition_set = set()
        for atoms in df["ase_atoms"]:
            composition_set.add(atoms.get_chemical_formula())

        # Choose a random sample of the unique compositions
        composition_list = list(composition_set)
        np.random.shuffle(composition_list)
        test_comps = composition_list[: int(len(composition_list) * train_test_fraction)]
        # Create a mask to filter the DataFrame
        mask = df["ase_atoms"].apply(lambda atoms: atoms.get_chemical_formula() in test_comps)
        # Filter the DataFrame
        df_new = [df[~mask], df[mask]]

    print(f"{len(df_new[0])} structures added to train set and {len(df_new[1])} structures added to test set")

    if database_paths:
        for ind, file in enumerate([database_paths["train"], database_paths["test"]]):
            df_old = pd.read_pickle(file, compression="gzip")
            df_concat = pd.concat([df_old] + [df_new[ind]])
            df_concat.to_pickle(file, compression="gzip", protocol=4)
    else:
        df_new[0].to_pickle(f"{wd}/train_data.pckl.gzip", compression="gzip", protocol=4)
        df_new[1].to_pickle(f"{wd}/test_data.pckl.gzip", compression="gzip", protocol=4)
        return {"train": f"{wd}/train_data.pckl.gzip", "test": f"{wd}/test_data.pckl.gzip"}
