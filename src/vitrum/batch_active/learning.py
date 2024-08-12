from vitrum.utility import get_random_packed
from vitrum.batch_active.flows import md_flow
from vitrum.batch_active.input_writer import lammps_input_writer
from vitrum.utility import apply_strain_to_structure
from itertools import product
from jobflow.managers.fireworks import flow_to_workflow
from jobflow import Flow
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
import yaml
from ase.io import read, write_lammps_data
import pandas as pd
import os
from fireworks import Firework, ScriptTask
from fireworks.core import Workflow
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor


class balace:
    def __init__(self, lp, units, mp_api_key, lammps_exe):
        self.state = "init"
        self.units = units
        self.atomtypes = [atom.symbol for atom in Composition("".join([unit for unit in units]))]
        self.mp_api_key = mp_api_key
        self.lp = lp
        self.runs = {}
        self.wd = os.getcwd()
        self.lammps_exe = lammps_exe

    def gen_even_structures(
        self,
        spacing: int = 10,
        datatype: str = "pymatgen",
    ) -> list:

        lists = [np.int32(np.linspace(0, 100, int(100 / spacing + 1))) for i in range(len(self.units))]
        all_combinations = product(*lists)
        valid_combinations = [combo for combo in all_combinations if sum(combo) == 100]
        structures = []
        for comb in valid_combinations:
            atoms_dict = {str(self.units[i]): comb[i] for i in range(len(self.units))}
            structures.append(
                get_random_packed(
                    atoms_dict, target_atoms=100, minAllowDis=1.7, mp_api_key=self.mp_api_key, datatype=datatype
                )
            )
        return structures

    def gen_strained_structures(self, structure, max_strain=0.2, num_strains=3):
        linear_strain = np.linspace(-max_strain, max_strain, num_strains)
        strain_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
        strained_structures = apply_strain_to_structure(structure, strain_matrices)
        return strained_structures, linear_strain

    def high_temp_run(self, structures=None, max_strain=0.2, num_strains=3, metadata=None):
        run_id = str(uuid.uuid4())
        if not structures:
            structures = self.gen_even_structures()
        flow_jobs = []
        for structure in structures:
            name = structure.reduced_formula
            if num_strains > 1:
                strained_structures, linear_strain = self.gen_strained_structures(structure, max_strain, num_strains)
                for strain, strain_struc in zip(linear_strain, strained_structures):
                    flow_jobs.append(md_flow(strain_struc, name=f"{name}_{strain}"))

            else:
                flow_jobs.append(md_flow(structure, name=name))

        flow = Flow(flow_jobs, name="MD_flows")
        wf = flow_to_workflow(flow, metadata={"uuid": run_id})
        self.runs.update({"high_temp_run": str(run_id)})
        return wf

    def get_atoms_from_wf(self, run_uuid):
        wf_id = [
            i
            for i in self.lp.get_wf_ids()
            if self.lp.get_wf_summary_dict(i, mode="all")["metadata"]["uuid"] == run_uuid
        ]
        atoms = []
        wf = self.lp.get_wf_summary_dict(wf_id)
        for fw in wf["states"]:
            if wf["states"][fw] == "COMPLETED":
                dirs = wf["launch_dirs"][fw][0]
                atoms_fw = read(f"{dirs}/OUTCAR.gz", format="vasp-out", index=":")
                atoms = atoms + atoms_fw
        return atoms

    def make_ace_database(self, atoms, iteration, force_threshold=100):
        energy = [i.get_total_energy() for i in atoms]
        force = [i.get_forces().tolist() for i in atoms]
        data = {"energy": energy, "forces": force, "ase_atoms": atoms, "iteration": iteration}
        # create a DataFrame
        df = pd.DataFrame(data)
        df = df[~df["forces"].apply(lambda x: np.max(x) > force_threshold)]
        return df

    def add_data_to_db(self, old_filename_db, new_filename_db, df_new_data: list):
        df_old = pd.read_pickle(old_filename_db, compression="gzip")
        df_new = pd.concat([df_old] + df_new_data)
        df_new.to_pickle(new_filename_db, compression="gzip", protocol=4)

    def save_dataframe(self, df):
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=1)
        df_train.to_pickle("train_data.pckl.gzip", compression="gzip", protocol=4)
        df_test.to_pickle("test_data.pckl.gzip", compression="gzip", protocol=4)

    def train_ace(self, iteration):
        with open("input.yaml") as f:
            list_doc = yaml.safe_load(f)
            list_doc["data"]["filename"] = f"{self.wd}/train_data.pckl.gzip"
            list_doc["data"]["test_filename"] = f"{self.wd}/test_data.pckl.gzip"

        os.makedirs(f"{self.wd}/ace/{iteration}")
        with open(f"{self.wd}/ace/{iteration}/input.yaml", "w") as f:
            yaml.dump(list_doc, f)

        firetask1 = ScriptTask.from_str(f"cd {self.wd}")
        firetask2 = ScriptTask.from_str("pacemaker input.yaml")

        fw = Firework([firetask1, firetask2], name="train_ace")
        self.lp.add_wf(fw)

    def get_structures_from_lammps(self):

        atoms_all = []
        for i in range(198):
            atoms = read(
                f"{folder}/structure_generation/structures/{i}_gamma_structures.dump",
                format="lammps-dump-text",
                index=":",
            )
            correct_chem_symbols(atoms)
            spaced_timesteps = get_spaced_timesteps(
                f"{folder}/structure_generation/structures/{i}_gamma_structures.dump", 100
            )
            atoms_all += [atoms_gamma[t] for t in spaced_timesteps]

        structures = [AseAtomsAdaptor().get_structure(atom) for atom in atoms_all]
        return structures

    def run_lammps(self, structures=None, max_strain=0.2, num_strains=3, metadata=None):
        run_id = str(uuid.uuid4())
        lammps_input_writer(
            self.wd,
            self.atomtypes,
            max_temp=5000,
            min_temp=300,
            cooling_rate=10,
            sample_rate=10000,
            seed=1,
            c_min=3,
            c_max=20,
        )
        os.makedirs(f"{self.wd}/gen_structures/{run_id}")
        if not structures:
            structures = self.gen_even_structures()

        fws = []
        for structure in structures:
            name = structure.reduced_formula
            if num_strains > 1:
                strained_structures, linear_strain = self.gen_strained_structures(structure, max_strain, num_strains)
                for strain, strain_struc in zip(linear_strain, strained_structures):
                    strain_struc = AseAtomsAdaptor().get_atoms(strain_struc)
                    os.makedirs(f"{self.wd}/gen_structures/{run_id}/{name}_{strain}")
                    write_lammps_data(
                        f"{self.wd}/gen_structures/{run_id}/{name}_{strain}/structure.dat",
                        strain_struc,
                        masses=True,
                        specorder=self.atom_types,
                    )
                    firetask1 = ScriptTask.from_str(f"cd {self.wd}/gen_structures/{run_id}/{name}_{strain}/")
                    firetask2 = ScriptTask.from_str(f"srun {self.exe} -in {self.wd}/in.ace")
                    fws.append(Firework([firetask1, firetask2], name=f"{name}_{strain}"))
            else:
                structure = AseAtomsAdaptor().get_atoms(structure)
                os.makedirs(f"{self.wd}/gen_structures/{run_id}/{name}")
                write_lammps_data(
                    f"{self.wd}/gen_structures/{run_id}/{name}/structure.dat",
                    structure,
                    masses=True,
                    specorder=self.atom_types,
                )
                firetask1 = ScriptTask.from_str(f"cd {self.wd}/gen_structures/{run_id}/{name}/")
                firetask2 = ScriptTask.from_str(f"srun {self.exe} -in {self.wd}/in.ace")
                fws.append(Firework([firetask1, firetask2], name=name))

        wf = Workflow(fws, name="lammps_runs", metadata={"uuid": run_id})
        self.lp.add_wf(wf)

    def run(self):
        if self.state == "init":
            wf = self.high_temp_run()
            self.state = "high_temp_run"
            self.lp.add_wf(wf)
            print("High temperature run added to workflow queue")

        if self.state == "high_temp_run":
            wf_id = self.runs["high_temp_run"]
            atoms = self.get_atoms_from_wf(wf_id)
            df = self.make_ace_database(atoms, self.iteration)
            self.save_dataframe(df)
            self.train_ace(self.iteration)
            self.iteration += 1
            print("Training ace model on high temperature data")
            self.state = "trained_ace"

        if self.state == "trained_ace":
            self.run_lammps()
            print("Generating structures with LAMMPS")
            self.state = "gen_lammps"

        if self.state == "gen_lammps":
            structures = self.get_structures_from_lammps()
            add_data_to_db
            self.train_ace(self.iteration)
            print(f"Training ace model: Iteration {self.iteration}")
            self.iteration += 1
