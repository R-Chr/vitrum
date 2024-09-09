from vitrum.utility import get_random_packed, get_LAMMPS_dump_timesteps
from vitrum.batch_active.flows import md_flow, static_flow
from vitrum.batch_active.input_writer import lammps_input_writer, ace_yaml_writer
from vitrum.utility import apply_strain_to_structure
from itertools import product
from jobflow.managers.fireworks import flow_to_workflow
from jobflow import Flow
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import pandas as pd
import os
from fireworks import Firework, ScriptTask, LaunchPad
from fireworks.core.firework import Workflow
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
import yaml
import pickle
from tqdm import tqdm
from fireworks.utilities.fw_serializers import load_object_from_file
from fireworks.queue.queue_launcher import rapidfire
from fireworks.core.fworker import FWorker
import subprocess


class balace:
    def __init__(self, config_file="balace.yaml", filename="balace.pickle", units=["SiO2"], auto_queue=False):
        self.auto_queue = auto_queue
        self.state = "high_temp_run"
        self.mp_api_key = None
        self.filename = filename
        self.runs = {}
        self.wd = os.getcwd()
        self.units = units
        self.iteration = 0

        if os.path.isfile(config_file) is False:
            raise FileNotFoundError(f"Config file {config_file} not found.")

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        for key, value in config.items():
            setattr(self, key, value)

        if not hasattr(self, "incar_settings"):
            self.incar_settings = False

        if not hasattr(self, "high_temp_params"):
            self.high_temp_params = {"temperature": 5000, "steps": 100}

        if not hasattr(self, "strain_params"):
            self.strain_params = {"num_strains": 3, "max_strain": 0.2}
        else:
            if "num_strains" not in self.strain_params:
                raise RuntimeError("strain_params is specified in yaml file but no num_strains specified")
            elif "max_strain" not in self.strain_params:
                raise RuntimeError("strain_params is specified in yaml file but no max_strain specified")

        self.atom_types = [atom.symbol for atom in Composition("".join([unit for unit in self.units]))]

        if hasattr(self, "launchpad"):
            self.lp = LaunchPad.from_file(self.launchpad)
        else:
            raise RuntimeError("Launchpad yaml not specified in config file.")

        if hasattr(self, "database"):
            if os.path.isfile(self.database["train"]) is False:
                raise FileNotFoundError("train_data.pckl.gzip not found")
            elif os.path.isfile(self.database["test"]) is False:
                raise FileNotFoundError("test_data.pckl.gzip not found")

        if hasattr(self, "qadapter_file"):
            self.qadapter = load_object_from_file(self.qadapter_file)

        if not hasattr(self, "reference_energy"):
            self.reference_energy = "auto"

        if not hasattr(self, "lammps_params"):
            self.lammps_params = None
        
        if not hasattr(self, "pace_params"):
            self.pace_params = None

    def save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self, f)

    def queue_rapidfire(self):
        rapidfire(self.lp, FWorker(), self.qadapter)

    def gen_even_structures(
        self,
        spacing: int = 10,
        datatype: str = "pymatgen",
    ) -> list:

        lists = [np.int32(np.linspace(0, 100, int(100 / spacing + 1))) for i in range(len(self.units))]
        all_combinations = product(*lists)
        valid_combinations = [combo for combo in all_combinations if sum(combo) == 100]
        structures = []
        for comb in tqdm(valid_combinations):
            atoms_dict = {str(self.units[i]): comb[i] for i in range(len(self.units))}
            structures.append(
                get_random_packed(
                    atoms_dict, target_atoms=100, minAllowDis=1.5, mp_api_key=self.mp_api_key, datatype=datatype
                )
            )
        return structures

    def gen_strained_structures(self, structure, max_strain=0.2, num_strains=3):
        linear_strain = np.linspace(-max_strain, max_strain, num_strains)
        strain_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        strained_structures = apply_strain_to_structure(structure, strain_matrices)
        struc = [strained_structures[index].final_structure for index in range(len(strain_matrices))]
        return struc, linear_strain

    def high_temp_run(self, structures=None):
        run_id = str(uuid.uuid4())
        if not structures:
            structures = self.gen_even_structures()
        flow_jobs = []
        for structure in structures:
            name = structure.reduced_formula
            if self.strain_params["num_strains"] > 1:
                strained_structures, linear_strain = self.gen_strained_structures(
                    structure, self.strain_params["max_strain"], self.strain_params["num_strains"]
                )
                for strain, strain_struc in zip(linear_strain, strained_structures):
                    flow_jobs.append(
                        md_flow(
                            strain_struc,
                            name=f"{name}_{strain}",
                            incar_settings=self.incar_settings,
                            temperature=self.high_temp_params["temperature"],
                            steps=self.high_temp_params["steps"],
                        )
                    )

            else:
                flow_jobs.append(
                    md_flow(
                        strain_struc,
                        name=f"{name}",
                        incar_settings=self.incar_settings,
                        temperature=self.high_temp_params["temperature"],
                        steps=self.high_temp_params["steps"],
                    )
                )
        flow = Flow(flow_jobs, name="MD_flows")
        wf = flow_to_workflow(flow, metadata={"uuid": run_id})
        if "DFT" not in self.runs:
            self.runs["DFT"] = [str(run_id)]
        else:
            self.runs["DFT"].append(str(run_id))
        return wf

    def get_atoms_from_wf(self, run_uuid, sampling=":"):
        wf_id = [
            i
            for i in self.lp.get_wf_ids()
            if self.lp.get_wf_summary_dict(i, mode="all")["metadata"]["uuid"] == run_uuid
        ]
        atoms = []
        wf = self.lp.get_wf_summary_dict(wf_id[0])
        for fw in wf["states"]:
            if wf["states"][fw] == "COMPLETED":
                dirs = wf["launch_dirs"][fw][0]
                atoms_fw = read(f"{dirs}/OUTCAR.gz", format="vasp-out", index=":")
                num_samples = len(atoms_fw)
                if sampling == ":":
                    atoms = atoms + atoms_fw
                elif isinstance(sampling, int):
                    sample_index = np.linspace(0, num_samples - 1, sampling, dtype=int)
                    atoms = atoms + [atoms_fw[i] for i in sample_index]
                elif isinstance(sampling, list):
                    atoms = atoms + [atoms_fw[i] for i in sampling]
        return atoms

    def update_ace_database(self, atoms, iteration, force_threshold=100):
        energy = [i.get_total_energy() for i in atoms]
        force = [i.get_forces().tolist() for i in atoms]
        data = {"energy": energy, "forces": force, "ase_atoms": atoms, "iteration": iteration}
        # create a DataFrame
        df = pd.DataFrame(data)
        df = df[~df["forces"].apply(lambda x: np.max(x) > force_threshold)]
        df_new = train_test_split(df, test_size=0.1, random_state=1)

        if hasattr(self, "database"):
            for ind, file in enumerate([self.database["train"], self.database["test"]]):
                df_old = pd.read_pickle(file, compression="gzip")
                df_concat = pd.concat([df_old] + df_new[ind])
                df_concat.to_pickle(file, compression="gzip", protocol=4)
        else:
            df_new[0].to_pickle(f"{self.wd}/train_data.pckl.gzip", compression="gzip", protocol=4)
            df_new[1].to_pickle(f"{self.wd}/test_data.pckl.gzip", compression="gzip", protocol=4)
            self.database = {"train": f"{self.wd}/train_data.pckl.gzip", "test": f"{self.wd}/test_data.pckl.gzip"}

    def train_ace(self):
        run_id = str(uuid.uuid4())
        directory = f"{self.wd}/ace_fitting/{run_id}"
        os.makedirs(f"{directory}")
        ace_yaml_writer(
            f"{directory}",
            self.database["train"],
            self.database["test"],
            self.atom_types,
            reference_energy=self.reference_energy,
        )
        firetask = ScriptTask.from_str(
            f"cd {directory} ; pacemaker input.yaml ; pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml"
        )
        wf = Workflow([Firework([firetask], name="training")], metadata={"uuid": run_id}, name="train_ace")
        self.lp.add_wf(wf)
        if "train_ace" not in self.runs:
            self.runs["train_ace"] = [directory]
        else:
            self.runs["train_ace"].append(directory)

    def run_lammps(self, structures=None, metadata=None):
        run_id = str(uuid.uuid4())
        lammps_input_writer(self.runs["train_ace"][-1], self.atom_types, **self.lammps_params)
        os.makedirs(f"{self.wd}/gen_structures/{run_id}")
        if not structures:
            structures = self.gen_even_structures()

        fws = []
        for structure in structures:
            name = structure.reduced_formula
            if self.strain_params["num_strains"] > 1:
                strained_structures, linear_strain = self.gen_strained_structures(
                    structure, self.strain_params["max_strain"], self.strain_params["num_strains"]
                )
                for strain, strain_struc in zip(linear_strain, strained_structures):
                    strain_struc = AseAtomsAdaptor().get_atoms(strain_struc)
                    os.makedirs(f"{self.wd}/gen_structures/{run_id}/{name}_{strain}")
                    write_lammps_data(
                        f"{self.wd}/gen_structures/{run_id}/{name}_{strain}/structure.dat",
                        strain_struc,
                        masses=True,
                        specorder=self.atom_types,
                    )
                    firetask = ScriptTask.from_str(
                        f"cd {self.wd}/gen_structures/{run_id}/{name}_{strain}/ ; srun {self.lammps_exe} -in {self.wd}/in.ace"
                    )
                    fws.append(Firework(firetask, name=f"{name}_{strain}"))
            else:
                structure = AseAtomsAdaptor().get_atoms(structure)
                os.makedirs(f"{self.wd}/gen_structures/{run_id}/{name}")
                write_lammps_data(
                    f"{self.wd}/gen_structures/{run_id}/{name}/structure.dat",
                    structure,
                    masses=True,
                    specorder=self.atom_types,
                )
                firetask = ScriptTask.from_str(
                    f"cd {self.wd}/gen_structures/{run_id}/{name}/ ; srun {self.lammps_exe} -in {self.wd}/in.ace"
                )
                fws.append(Firework(firetask, name=name))

        wf = Workflow(fws, name="lammps_runs", metadata={"uuid": run_id})
        self.lp.add_wf(wf)

        if "run_lammps" not in self.runs:
            self.runs["run_lammps"] = [run_id]
        else:
            self.runs["run_lammps"].append(run_id)

    def correct_chem_symbols(self, atom_types, atoms):
        symbol_change_map = {i + 1: x for i, x in enumerate(atom_types)}
        for atom in atoms:
            chem_symbols = [symbol_change_map.get(x, x) for x in atom.get_atomic_numbers()]
            atom.set_chemical_symbols(chem_symbols)

    def get_structures_from_lammps(self, pace_select=True, force_glass_structures=True, use_spaced_timesteps=False):
        folder = f"{self.wd}/gen_structures/{self.runs['run_lammps'][-1]}"
        
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
                                select_files.append(file_path)
                        else:
                            select_files.append(file_path)
                    else:
                        forced_files.append(file_path)

        atoms_selected = []
        atoms_forced = []

        if pace_select is True:
            atoms_selected += self.select_structures(select_files, **self.pace_params)
        
        for file_path in forced_files:
            atoms = read(file_path, format="lammps-dump-text", index=":")
            if len(atoms) == 0:
                continue
            self.correct_chem_symbols(self.atom_types, atoms)
            if use_spaced_timesteps is True:
                timesteps = get_LAMMPS_dump_timesteps(file_path)
                spaced_timesteps = [0]
                for ind, time in enumerate(timesteps):
                    if time > timesteps[spaced_timesteps[-1]] + 100:
                        spaced_timesteps.append(ind)
                atoms_forced += [atoms[t] for t in spaced_timesteps]
            else:
                atoms_forced += atoms

        structures = [AseAtomsAdaptor().get_structure(atom) for atom in atoms_forced] + [AseAtomsAdaptor().get_structure(atom) for atom in atoms_selected]

        return structures

    def static_run(self, structures, metadata=None):
        run_id = str(uuid.uuid4())
        flow_jobs = []
        for structure in structures:
            name = structure.reduced_formula
            flow_jobs.append(static_flow(structure, name=name, incar_settings=self.incar_settings))

        flow = Flow(flow_jobs, name="Static_flows")
        wf = flow_to_workflow(flow, metadata={"uuid": run_id})
        if "DFT" not in self.runs:
            self.runs["DFT"] = [str(run_id)]
        else:
            self.runs["DFT"].append(str(run_id))
        return wf

    def select_structures(self, select_files, num_select_structures=500):
        atom_string = " ".join([str(atom) for atom in self.atom_types])
        file_string = " ".join(select_files)
        subprocess.run(
            f"pace_select -p output_potential.yaml -a output_potential.asi -e {atom_string} -m {num_select_structures} {file_string}}"
        )
        atoms = pd.read_pickle(f'selected.pkl.gz', compression='gzip')
        return [structure for structure in atoms["ase_atoms"]]

    def run(self):
        if self.state == "high_temp_run":
            wf = self.high_temp_run()
            self.state = "train_ace"
            self.lp.add_wf(wf)
            print("High temperature run added to workflow queue")

        elif self.state == "train_ace":
            wf_id = self.runs["DFT"][-1]
            if self.iteration > 0:
                atoms = self.get_atoms_from_wf(wf_id, sampling=":")
            else:
                atoms = self.get_atoms_from_wf(wf_id, sampling=5)

            self.update_ace_database(atoms, self.iteration)
            self.train_ace()
            print(f"Training ace model, Iteration: {self.iteration}")
            self.iteration += 1
            self.state = "gen_lammps"

        elif self.state == "gen_lammps":
            self.run_lammps()
            print("Generating structures with LAMMPS")
            self.state = "evaluate"

        elif self.state == "evaluate":
            structures = self.get_structures_from_lammps()
            selected_structures = self.select_structures(structures)
            wf = self.static_run(structures)
            self.lp.add_wf(wf)
            self.state = "train_ace"
            print("Evaluating new structures with VASP")

        self.save()

        if self.auto_queue is True:
            self.queue_rapidfire() if hasattr(self, "qadapter") else print("No qadapter found, skipping auto queue")
