from vitrum.utility import get_random_packed
from vitrum.batch_active.flows import strained_flows
from itertools import product
from jobflow.managers.fireworks import flow_to_workflow
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
import yaml
from ase.io import read
import pandas as pd
import os
from fireworks import Firework, FWorker, LaunchPad, ScriptTask, TemplateWriterTask, FileTransferTask


class balace:
    def __init__(self, lp, units, mp_api_key):
        self.units = units
        self.mp_api_key = mp_api_key
        self.lp = lp
        self.runs = {}
        self.wd = os.getcwd()

    def gen_even_structures(
        self,
        spacing: int = 10,
    ) -> list:

        lists = [np.int32(np.linspace(0, 100, int(100 / spacing + 1))) for i in range(len(self.units))]
        all_combinations = product(*lists)
        valid_combinations = [combo for combo in all_combinations if sum(combo) == 100]
        structures = []
        for comb in valid_combinations:
            atoms_dict = {str(self.units[i]): comb[i] for i in range(len(self.units))}
            structures.append(
                get_random_packed(
                    atoms_dict, target_atoms=100, minAllowDis=1.7, mp_api_key=self.mp_api_key, datatype="pymatgen"
                )
            )
        return structures

    def high_temp_run(self):
        run_id = str(uuid.uuid4())
        structures = self.gen_even_structures()
        flow = strained_flows(structures, metadata=run_id)
        wf = flow_to_workflow(flow, metadata={"uuid": run_id})
        self.lp.add_wf(wf)
        self.runs.update({"high_temp_run": str(run_id)})

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
