import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def update_ace_database(wd, atoms, iteration, force_threshold=100, database_paths=None, metadata=None):
    energy = [i.get_total_energy() for i in atoms]
    force = [i.get_forces().tolist() for i in atoms]
    data = {"energy": energy, "forces": force, "ase_atoms": atoms, "iteration": iteration}
    if metadata:
        data["sample_type"] = metadata
    # create a DataFrame
    df = pd.DataFrame(data)
    print(f"Iteration {iteration} has {len(df)} structures")
    df = df[~df["forces"].apply(lambda x: np.max(x) > force_threshold)]
    df = df[~df["forces"].apply(lambda x: np.min(x) < -force_threshold)]
    print(f"{len(df)} structures remain after force threshold filter")
    df_new = train_test_split(df, test_size=0.1, random_state=1)
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
