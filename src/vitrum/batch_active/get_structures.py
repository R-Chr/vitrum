from ase.io import read
import numpy as np
import os
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
    wf_ids = [get_wflow_id_from_run_uuid(id) for id in run_uuids]
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
            if fw.states == "COMPLETED":
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
    selection_params=None,
    pace_select=True,
    force_glass_structures=True,
    use_spaced_timesteps=False,
    **kwargs,
):
    select_files = []
    forced_files = []
    for dirpath, _, filenames in os.walk(folder):
        for file in ["glass.dump", "gamma.dump"]:
            if file in filenames:
                file_path = os.path.join(dirpath, file)
                if pace_select is True:
                    if force_glass_structures is True:
                        if file == "glass.dump":
                            forced_files.append(file_path)
                        else:
                            select_files.append(file_path.replace(")", r"\)").replace("(", r"\("))
                    else:
                        select_files.append(file_path.replace(")", r"\)").replace("(", r"\("))
                else:
                    forced_files.append(file_path)

    atoms_selected = []
    atoms_forced = []

    if pace_select is True:
        print("Running PACE select")
        atoms_selected += select_structures(potential_folder, atom_types, select_files, **selection_params)

    for file_path in forced_files:
        atoms = read(file_path, format="lammps-dump-text", index=":")
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


def select_structures(folder, atom_types, select_files, num_select_structures=500, **kwargs):
    atom_string = " ".join([str(atom) for atom in atom_types])
    file_string = " ".join(select_files)
    subprocess.run(
        f"pace_select -p {folder}/output_potential.yaml -a "
        f'{folder}/output_potential.asi -e "{atom_string}"'
        f" -m {num_select_structures} {file_string}",
        shell=True,
    )
    atoms = pd.read_pickle("selected.pkl.gz", compression="gzip")
    return [structure for structure in atoms["ase_atoms"]]
