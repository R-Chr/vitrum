from ase.io import read
import numpy as np
from pathlib import Path
from vitrum.utility import get_LAMMPS_dump_timesteps, correct_atom_types
import subprocess
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd


def get_wflow_id_from_run_uuid(lp, run_uuid):
    wf_ids = [i for i in lp.get_wf_ids() if lp.get_wf_summary_dict(i, mode="all")["metadata"]["uuid"] == run_uuid][0]
    return wf_ids


def get_atoms_from_wfs(lp, run_uuids, high_temp_params, sampling=":", state=None):
    """
    Reads all atoms from a workflow given by uuid and returns them.

    Parameters:
        run_uuids : list
            list of uuids of the workflows to read from.
        sampling : str or list or int, optional
            If sampling is a string, it is interpreted as a slice string for numpy.
            If it is an integer, it is interpreted as the number of samples to take.
            If it is a list, it is interpreted as a list of indices to sample.
            Defaults to ":".

    Returns:
        atoms: list
            A list of ase atoms objects.
    """
    wf_ids = [get_wflow_id_from_run_uuid(lp, id) for id in run_uuids]
    atoms = []
    metadata = []

    if state == "train_ace_high_temp":
        sampling = high_temp_params["sampling"]
    else:
        sampling = sampling

    for wf_id in wf_ids:
        wf = lp.get_wf_by_fw_id(wf_id)
        launch_dirs = [fw.launches[0].launch_dir if fw.launches else None for fw in wf.fws]
        for dirs, fw in zip(launch_dirs, wf.fws):
            if fw.state == "COMPLETED":
                atoms_fw = read(f"{dirs}/OUTCAR.gz", format="vasp-out", index=":")
                num_samples = len(atoms_fw)
                if sampling == ":":
                    atoms = atoms + atoms_fw
                    num_samples = len(atoms_fw)
                elif isinstance(sampling, int):
                    sample_index = np.linspace(0, num_samples - 1, sampling, dtype=int)
                    atoms = atoms + [atoms_fw[i] for i in sample_index]
                    num_samples = len(sample_index)
                elif isinstance(sampling, list):
                    atoms = atoms + [atoms_fw[i] for i in sampling]
                    num_samples = len(sampling)
                metadata = metadata + [fw.spec["sample_type"]] * num_samples

    return atoms, metadata


def get_structures_from_lammps(
    folder,
    potential_folder,
    atom_types,
    potential,
    pace_select=True,
    force_glass_structures=True,
    use_spaced_timesteps=False,
    max_gamma_structures=500,
):
    select_files = []
    forced_files = []

    folder_path = Path(folder)
    for dirpath in folder_path.rglob("*"):  # Recursively iterate over all directories/files
        if dirpath.is_dir():  # Ensure it's a directory
            for file in ["glass.dump", "gamma.dump"]:
                file_path = dirpath / file  # Use pathlib's `/` operator to join paths
                if file_path.exists():  # Check if file exists
                    file_path_str = str(file_path).replace(")", r"\)").replace("(", r"\(")

                    if pace_select:
                        if force_glass_structures:
                            if file == "glass.dump":
                                forced_files.append(file_path_str)
                            else:
                                select_files.append(file_path_str)
                        else:
                            select_files.append(file_path_str)
                    else:
                        forced_files.append(file_path_str)

    atoms_selected = []
    atoms_forced = []

    if pace_select is True:
        print("Running PACE select")
        atoms_selected += select_structures(
            potential_folder, atom_types, select_files, potential, num_select_structures=max_gamma_structures
        )

    for file_path in forced_files:
        atoms = read(file_path.replace("\\", ""), format="lammps-dump-text", index=":")
        if len(atoms) == 0:
            continue
        symbol_change_map = {i + 1: x for i, x in enumerate(atom_types)}
        atoms = correct_atom_types(atoms, symbol_change_map)

        if use_spaced_timesteps is True:
            timesteps = get_LAMMPS_dump_timesteps(file_path)
            spaced_timesteps = [0]
            for ind, time in enumerate(timesteps):
                if time > timesteps[spaced_timesteps[-1]] + 100:
                    spaced_timesteps.append(ind)
            atoms_forced += [atoms[t] for t in spaced_timesteps]
        else:
            atoms_forced += atoms

    print(f"Included {len(atoms_selected)} selected structures and {len(atoms_forced)} forced structures.")
    metadata = ["manual"] * len(atoms_selected) + ["gamma"] * len(atoms_forced)
    structures = [AseAtomsAdaptor().get_structure(atom) for atom in atoms_forced] + [
        AseAtomsAdaptor().get_structure(atom) for atom in atoms_selected
    ]

    return structures, metadata


def select_structures(folder, atom_types, select_files, potential, num_select_structures=500):
    print(select_files)
    atom_string = " ".join([str(atom) for atom in atom_types])
    file_string = " ".join(select_files)
    if potential == "pace":
        subprocess.run(
            f"pace_select -p {folder}/output_potential.yaml -a "
            f'{folder}/output_potential.asi -e "{atom_string}"'
            f" -m {num_select_structures} {file_string}",
            shell=True,
        )
    elif potential == "grace":
        subprocess.run(
            f"pace_select -p {folder}/FS_model.yaml"
            f' -a {folder}/FS_model.asi -e "{atom_string}"'
            f" -m {num_select_structures} {file_string}",
            shell=True,
        )
    atoms = pd.read_pickle("selected.pkl.gz", compression="gzip")
    return [structure for structure in atoms["ase_atoms"]]
