from vitrum.batch_active.input_writer import lammps_input_writer
from vitrum.batch_active.structure_gen import gen_even_structures, gen_lammps_structures
from vitrum.batch_active.workflow import high_temp_run, static_run, train_pace, run_lammps, rerun_crashed_jobs
from vitrum.batch_active.database import update_ace_database
from vitrum.batch_active.get_structures import (
    get_atoms_from_wfs,
    get_wflow_id_from_run_uuid,
    get_structures_from_lammps,
)
import uuid
import os
from fireworks import LaunchPad
from pymatgen.core import Composition

import yaml
import pickle
from fireworks.utilities.fw_serializers import load_object_from_file
from fireworks.queue.queue_launcher import rapidfire
from fireworks.core.fworker import FWorker


class balace:
    def __init__(self, config_file="balace.yaml", filename="balace.pickle", units=["SiO2"], auto_queue=False):
        """
        Initialize the balace class.

        Parameters:
            config_file (str): yaml file containing configuration for the balace class. Defaults to "balace.yaml".
            filename (str): filename to save the class to. Defaults to "balace.pickle".
            units (list of str): list of composition units to use. Defaults to ["SiO2"].
            auto_queue (bool): whether to automatically queue runs. Defaults to False.

        Attributes:
            auto_queue (bool): whether to automatically queue runs
            state (str): current state of the balace run
            mp_api_key (str): materials project API keys
            filename (str): filename to save the class to
            runs (dict): dictionary containing information about runs
            wd (str): working directory
            units (list of str): list of composition units to use
            iteration (int): current iteration number
            atom_types (list of str): list of atom types
            incar_settings (dict): dictionary containing incar settings
            high_temp_params (dict): dictionary containing high temperature parameters
            strain_params (dict): dictionary containing strain parameters
            launchpad (str): launchpad yaml file
            database (dict): dictionary containing database information
            qadapter_file (str): file containing qadapter
            reference_energy (str): reference energy to use
            lammps_params (dict): dictionary containing lammps parameters
            selection_params (dict): dictionary containing selection parameters
            composition_params (dict): dictionary containing composition parameters
        """
        self.auto_queue = auto_queue
        self.state = "high_temp_run"
        self.filename = filename
        self.runs = {}
        self.wd = os.getcwd()
        self.units = units
        self.iteration = 0
        self.config_file = config_file
        self.read_config()
        self.atom_types = [atom.symbol for atom in Composition("".join([unit for unit in self.units]))]

    def read_config(self):
        if os.path.isfile(self.config_file) is False:
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        with open(self.config_file, "r") as file:
            config = yaml.safe_load(file)

        for key, value in config.items():
            setattr(self, key, value)

        if not hasattr(self, "mp_api_key"):
            raise RuntimeError("mp_api_key not specified in config file.")

        if hasattr(self, "launchpad"):
            self.lp = LaunchPad.from_file(self.launchpad)
        else:
            raise RuntimeError("Launchpad yaml not specified in config file.")

        if not hasattr(self, "incar_settings"):
            self.incar_settings = False

        if not hasattr(self, "high_temp_params"):
            self.high_temp_params = {"temperature": 5000, "steps": 100, "sampling": 5}

        if not hasattr(self, "strain_params"):
            self.strain_params = {"num_strains": 3, "max_strain": 0.2}
        else:
            if "num_strains" not in self.strain_params:
                raise RuntimeError("strain_params is specified in yaml file but no num_strains specified")
            elif "max_strain" not in self.strain_params:
                raise RuntimeError("strain_params is specified in yaml file but no max_strain specified")

        if hasattr(self, "database"):
            if os.path.isfile(self.database["train"]) is False:
                raise FileNotFoundError("train_data.pckl.gzip not found")
            elif os.path.isfile(self.database["test"]) is False:
                raise FileNotFoundError("test_data.pckl.gzip not found")
        else:
            self.database = None

        if hasattr(self, "qadapter_file"):
            self.qadapter = load_object_from_file(self.qadapter_file)

        if not hasattr(self, "reference_energy"):
            self.reference_energy = "auto"

        if not hasattr(self, "lammps_params"):
            self.lammps_params = {}

        if not hasattr(self, "selection_params"):
            self.selection_params = {}

        if not hasattr(self, "composition_params"):
            self.composition_params = {}

    def save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self, f)

    def queue_rapidfire(self):
        rapidfire(self.lp, FWorker(), self.qadapter)

    def check_resubmit_high_temp(self, run_uuids):
        run_id = str(uuid.uuid4())
        wf_id = get_wflow_id_from_run_uuid(self.lp, run_uuids[-1])
        wf = self.lp.get_wf_by_fw_id(wf_id)
        if "READY" in wf.fw_states.values():
            raise ValueError(f"READY jobs in wf: {wf_id} needs to be completed before proceeding")
        if "RUNNING" in wf.fw_states.values():
            raise ValueError(f"RUNNING jobs in wf: {wf_id} needs to be completed before proceeding")

        crashed_jobs = []
        fw_ids = [fw_id for fw_id, state in wf.fw_states.items() if state == "FIZZLED"]
        for id in fw_ids:
            dic = self.lp.get_fw_by_id(id)
            crashed_jobs.append({"composition": dic.spec["composition"], "strain": dic.spec["strain"]})

        if len(crashed_jobs) == 0:
            print(f"No crashed jobs in wf: {wf_id}")
            print("All jobs completed successfully, no resubmission needed.")
            return False

        else:
            print(f"Found {len(crashed_jobs)} crashed jobs in wf: {wf_id}")
            print("Resubmitting crashed jobs...")
            wf = rerun_crashed_jobs(crashed_jobs, run_id, self.incar_settings, self.high_temp_params, self.mp_api_key)
            self.runs["DFT"][-1].append(str(run_id))
            self.lp.add_wf(wf)
            return True

    def run_pace(self):
        """
        The main loop for the active learning workflow. It runs through the following states:

        - high_temp_run: Runs a high temperature AIMD simulation using VASP.
        - train_ace: Train the ACE model using the current database of structures.
        - gen_lammps: Runs a LAMMPS simulation with the current ACE potential to generate new structures.
        - evaluate: Evaluates the new structures with VASP.

        Parameters:
            None

        Returns:
            None
        """
        print(f"Current state: {self.state}, Iteration: {self.iteration}")

        if self.state == "high_temp_run":
            structures = gen_even_structures(units=self.units, mp_api_key=self.mp_api_key, **self.composition_params)
            wf, run_id = high_temp_run(structures, self.strain_params, self.incar_settings, self.high_temp_params)
            if "DFT" not in self.runs:
                self.runs["DFT"] = [[str(run_id)]]
            else:
                self.runs["DFT"].append([str(run_id)])
            self.state = "train_ace_high_temp"
            self.lp.add_wf(wf)
            print("High temperature run added to workflow queue")

        elif self.state == "train_ace_high_temp" or self.state == "train_ace_lammps":
            previous_run_ids = self.runs["DFT"][-1]
            if self.state == "train_ace_high_temp":
                stop = self.check_resubmit_high_temp(previous_run_ids)
            else:
                stop = False

            if stop:
                pass
            else:
                atoms = get_atoms_from_wfs(self.lp, previous_run_ids, self.high_temp_params, state=self.state)
                self.database = update_ace_database(self.wd, atoms, self.iteration, database_paths=self.database)
                wf, directory = train_pace()
                self.lp.add_wf(wf)
                if "train_ace" not in self.runs:
                    self.runs["train_ace"] = [directory]
                else:
                    self.runs["train_ace"].append(directory)

                print(f"Training ace model, Iteration: {self.iteration}")
                self.iteration += 1
                self.state = "gen_lammps"

        elif self.state == "gen_lammps":
            pot_dir = self.runs["train_ace"][-1]
            if not os.path.exists(f"{pot_dir}/output_potential.asi"):
                raise FileNotFoundError(
                    f"{pot_dir}/output_potential.asi not found. Make sure ACE potential has completed training"
                )

            run_id = str(uuid.uuid4())
            print(f"Setting up for LAMMPS runs in /gen_structures/{run_id}")
            lammps_input_writer(self.runs["train_ace"][-1], self.atom_types, **self.lammps_params)
            path = f"{self.wd}/gen_structures/{run_id}"
            os.makedirs(path)
            structures = gen_even_structures(units=self.units, mp_api_key=self.mp_api_key, **self.composition_params)
            directories = gen_lammps_structures(structures, self.strain_params, specorder=self.atom_types, path=path)
            wfs = run_lammps(directories)
            self.lp.add_wf(wfs)
            if "run_lammps" not in self.runs:
                self.runs["run_lammps"] = [run_id]
            else:
                self.runs["run_lammps"].append(run_id)
            print("Generating structures with LAMMPS")
            self.state = "evaluate"

        elif self.state == "evaluate":
            folder = f"{self.wd}/gen_structures/{self.runs['run_lammps'][-1]}"
            potential_folder = self.runs["train_ace"][-1]
            structures = get_structures_from_lammps(folder, potential_folder, **self.selection_params)
            wf, run_id = static_run(structures)
            if "DFT" not in self.runs:
                self.runs["DFT"] = [[str(run_id)]]
            else:
                self.runs["DFT"].append([str(run_id)])
            self.lp.add_wf(wf)
            self.state = "train_ace_lammps"
            print("Evaluating new structures with VASP")

        self.save()

        if self.auto_queue is True:
            self.queue_rapidfire() if hasattr(self, "qadapter") else print("No qadapter found, skipping auto queue")
