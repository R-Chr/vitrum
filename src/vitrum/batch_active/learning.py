from vitrum.batch_active.input_writer import lammps_input_writer
from vitrum.batch_active.structure_gen import gen_even_structures, gen_lammps_structures
from vitrum.batch_active.workflow import high_temp_run, static_run, train_pace, run_lammps, rerun_crashed_jobs
from vitrum.batch_active.database import update_ace_database
from vitrum.batch_active.get_structures import (
    get_atoms_from_wfs,
    get_wflow_id_from_run_uuid,
    get_structures_from_lammps,
)

from vitrum.structure_gen import gen_random_glasses
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
    def __init__(self, config_file="balace.yaml", filename="balace.pickle", auto_queue=False):
        """
        Initialize the balace class.

        Parameters:
            config_file (str): yaml file containing configuration for the balace class. Defaults to "balace.yaml".
            filename (str): filename to save the class to. Defaults to "balace.pickle".
            auto_queue (bool): whether to automatically queue runs. Defaults to False.

        Attributes:
            auto_queue (bool): whether to automatically queue runs
            state (str): current state of the balace run
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
        self.state = "start"
        self.filename = filename
        self.runs = {}
        self.wd = os.getcwd()
        self.iteration = 0
        self.config_file = config_file
        self.load_config()
        self.set_defaults()
        self.validate_config()
        if self.struc_gen_params["scheme"] == "even":
            self.atom_types = [
                atom.symbol for atom in Composition("".join([unit for unit in self.struc_gen_params["units"]]))
            ]
        elif self.struc_gen_params["scheme"] == "random":
            self.atom_types = (
                self.struc_gen_params["atoms"]["modifiers"]
                + self.struc_gen_params["atoms"]["formers"]
                + self.struc_gen_params["atoms"]["anions"]
            )

    def load_config(self):
        """Loads the YAML configuration file."""
        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        with open(self.config_file, "r") as file:
            self.config = yaml.safe_load(file)

        # Apply config values as attributes
        for key, value in self.config.items():
            setattr(self, key, value)

    def set_defaults(self):
        """Sets default values for missing attributes."""
        self.incar_settings = getattr(self, "incar_settings", False)
        self.high_temp_params = getattr(self, "high_temp_params", {"temperature": 5000, "steps": 100, "sampling": 5})
        self.strain_params = getattr(self, "strain_params", {"num_strains": 3, "max_strain": 0.2})
        self.database = getattr(self, "database", None)
        self.qadapter = load_object_from_file(self.qadapter_file) if hasattr(self, "qadapter_file") else None
        self.reference_energy = getattr(self, "reference_energy", "auto")
        self.lammps_params = getattr(self, "lammps_params", {})
        self.selection_params = getattr(self, "selection_params", {})
        self.composition_params = getattr(self, "composition_params", {})
        self.struc_gen_params = getattr(
            self, "struc_gen_params", {"scheme": "even", "units": ["SiO2"], "target_atoms": 100}
        )

    def validate_config(self):
        """Validates required config values."""
        if not hasattr(self, "struc_gen_params"):
            raise RuntimeError("struc_gen_params are not specified in config file.")

        if self.struc_gen_params["scheme"] == "even" and "units" not in self.struc_gen_params:
            raise RuntimeError("units must be specified in struc_gen_params for even scheme.")

        if self.struc_gen_params["scheme"] == "random" and "atoms" not in self.struc_gen_params:
            raise RuntimeError("atoms must be specified in struc_gen_params for random scheme.")

        if not hasattr(self, "potential"):
            raise RuntimeError("potential type must be specified in config file, e.g. pace or grace.")

        if not hasattr(self, "lammps_command"):
            raise RuntimeError("lammps_command not specified in config file.")

        if not hasattr(self, "launchpad"):
            raise RuntimeError("Launchpad yaml not specified in config file.")
        self.lp = LaunchPad.from_file(self.launchpad)

        if "num_strains" not in self.strain_params or "max_strain" not in self.strain_params:
            raise RuntimeError("strain_params is missing required fields (num_strains, max_strain).")

        if self.database:
            if not os.path.isfile(self.database.get("train", "")):
                raise FileNotFoundError("train_data.pckl.gzip not found")
            if not os.path.isfile(self.database.get("test", "")):
                raise FileNotFoundError("test_data.pckl.gzip not found")

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
            wf = rerun_crashed_jobs(crashed_jobs, run_id, self.incar_settings, self.high_temp_params)
            self.runs["DFT"][-1].append(str(run_id))
            self.lp.add_wf(wf)
            return True

    def generate_structures(self):
        if self.struc_gen_params["scheme"] == "even":
            structures = gen_even_structures(
                units=self.self.struc_gen_params["units"],
                target_atoms=self.struc_gen_params["target_atoms"],
                **self.composition_params,
            )
        elif self.struc_gen_params["scheme"] == "random":
            atoms = self.struc_gen_params["atoms"]
            structures = gen_random_glasses(
                atoms["modifiers"],
                atoms["formers"],
                atoms["anions"],
                weights=self.struc_gen_params.get("weights", {}),
                num_structures=self.struc_gen_params["num_structures"],
                target_atoms=self.struc_gen_params["target_atoms"],
                datatype="pymatgen",
            )
        return structures

    def run_high_temp(self):
        structures = self.generate_structures()
        wf, run_id = high_temp_run(structures, self.strain_params, self.incar_settings, self.high_temp_params)
        self.runs.setdefault("DFT", []).append([str(run_id)])
        self.state = "high_temp_AIMD"
        self.lp.add_wf(wf)
        print("High temperature run added to workflow queue")

    def run_train_pace(self):
        previous_run_ids = self.runs["DFT"][-1]
        if self.state == "high_temp_AIMD":
            stop = self.check_resubmit_high_temp(previous_run_ids)
        else:
            stop = False

        if not stop:
            atoms, metadata = get_atoms_from_wfs(self.lp, previous_run_ids, self.high_temp_params, state=self.state)
            self.database = update_ace_database(
                self.wd, atoms, self.iteration, database_paths=self.database, metadata=metadata
            )
            wf, directory = train_pace()
            self.lp.add_wf(wf)
            self.runs.setdefault("potential", []).append(directory)
            print(f"Training ace model, Iteration: {self.iteration}")
            self.iteration += 1
            self.state = "trained_ace"

    def run_train_grace(self):
        previous_run_ids = self.runs["DFT"][-1]
        atoms, metadata = get_atoms_from_wfs(self.lp, previous_run_ids, self.high_temp_params, state=self.state)
        self.database = update_ace_database(
            self.wd, atoms, self.iteration, database_paths=self.database, metadata=metadata
        )
        print(f"Train Grace based on dataset, Iteration: {self.iteration}")
        self.iteration += 1
        self.state = "trained_ace"

    def run_gen_lammps(self):
        pot_dir = self.runs["potential"][-1]

        if self.potential == "pace" and not os.path.exists(f"{pot_dir}/output_potential.yaml"):
            raise FileNotFoundError(
                f"{pot_dir}/output_potential.yaml not found. Make sure ACE potential has completed training"
            )

        run_id = str(uuid.uuid4())
        print(f"Setting up for LAMMPS runs in /gen_structures/{run_id}")
        lammps_input_writer(self.runs["potential"][-1], self.potential, self.atom_types, **self.lammps_params)
        path = f"{self.wd}/gen_structures/{run_id}"
        os.makedirs(path)
        structures = self.generate_structures()
        directories = gen_lammps_structures(structures, self.strain_params, specorder=self.atom_types, path=path)
        wfs = run_lammps(directories, self.wd, self.lammps_command, run_id)
        self.lp.add_wf(wfs)
        self.runs.setdefault("run_lammps", []).append(run_id)
        print("Generating structures with LAMMPS")
        self.state = "lammps_runs"

    def run_evaluate(self):
        folder = f"{self.wd}/gen_structures/{self.runs['run_lammps'][-1]}"
        potential_folder = self.runs["potential"][-1]
        structures, metadata = get_structures_from_lammps(folder, potential_folder, **self.selection_params)
        wf, run_id = static_run(structures, metadata)
        self.runs.setdefault("DFT", []).append([str(run_id)])
        self.lp.add_wf(wf)
        self.state = "static_runs"
        print("Evaluating new structures with VASP")

    def run_pace(self):
        """
        The main loop for the active learning workflow. It runs through the following states:
        - Runs a high temperature AIMD simulation using VASP.
        - Trains the ACE model using the current database of structures.
        - Runs LAMMPS simulations with the current ACE potential to generate new structures.
        - Evaluates the new structures with VASP static calculations.

        Parameters:
            None

        Returns:
            None
        """
        print(f"Current state: {self.state}, Iteration: {self.iteration}")

        # Mapping states to methods
        state_methods = {
            "start": self.run_high_temp,
            "high_temp_AIMD": self.run_train_pace,
            "static_runs": self.run_train_pace,
            "trained_ace": self.run_gen_lammps,
            "lammps_runs": self.run_evaluate,
        }

        # Get the corresponding method and execute it
        state_method = state_methods.get(self.state)
        if state_method:
            state_method()
        else:
            raise ValueError(f"Unknown state: {self.state}")

        self.save()

        if self.auto_queue is True:
            self.queue_rapidfire() if hasattr(self, "qadapter") else print("No qadapter found, skipping auto queue")

    def run_grace(self):
        print(f"Current state: {self.state}, Iteration: {self.iteration}")
        if self.state == "start":
            if not hasattr(self, "initial_potential"):
                raise ValueError("Initial potential not specified in config file.")
            directory = self.initial_potential
            self.runs.setdefault("potential", []).append(directory)

        # Mapping states to methods
        state_methods = {
            "start": self.run_gen_lammps,
            "static_runs": self.run_train_grace,
            "trained_ace": self.run_gen_lammps,
            "lammps_runs": self.run_evaluate,
        }

        # Get the corresponding method and execute it
        state_method = state_methods.get(self.state)
        if state_method:
            state_method()
        else:
            raise ValueError(f"Unknown state: {self.state}")

        self.save()

    def run(self):
        if self.potential == "pace":
            self.run_pace()
        elif self.potential == "grace":
            self.run_grace()
        else:
            raise ValueError(f"Unsupported potential type: {self.potential}")
