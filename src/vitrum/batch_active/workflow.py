from vitrum.batch_active.structure_gen import gen_strained_structures
import uuid
from vitrum.batch_active.flows import md_flow, static_flow
from jobflow.managers.fireworks import flow_to_workflow
from jobflow import Flow
import os
from vitrum.batch_active.input_writer import ace_yaml_writer
from fireworks import Firework, ScriptTask
from fireworks.core.firework import Workflow
from vitrum.utility import get_random_packed, apply_strain_to_structure
import numpy as np


def high_temp_run(structures, strain_params, incar_settings, high_temp_params):
    """
    Run a series of high temperature MD simulations on a list of structures.

    Parameters:
        structures: list, optional
            List of structures to run the simulations on. If not given, will use the
            composition parameters to generate a list of structures.

    Returns:
    wf: Workflow
        The workflow to run the simulations
    """
    run_id = str(uuid.uuid4())

    flow_jobs = []
    for structure in structures:
        composition = structure.reduced_formula
        strained_structures, linear_strain = gen_strained_structures(
            structure, strain_params["max_strain"], strain_params["num_strains"]
        )
        for strain, strain_struc in zip(linear_strain, strained_structures):
            job = md_flow(
                strain_struc,
                name=f"{composition}_{strain}",
                incar_settings=incar_settings,
                temperature=high_temp_params["temperature"],
                steps=high_temp_params["steps"],
            )
            job.update_metadata({"strain": strain, "composition": composition, "sample_type": "high_temp"})
            flow_jobs.append(job)

    flow = Flow(flow_jobs, name="MD_flows")
    wf = flow_to_workflow(flow, metadata={"uuid": run_id})

    return wf, run_id


def static_run(structures, incar_settings, metadata=None):
    run_id = str(uuid.uuid4())
    wfs = []
    if metadata is None:
        metadata = [None] * len(structures)

    for structure, m_data in zip(structures, metadata):
        name = structure.reduced_formula
        job = static_flow(structure, name=name, incar_settings=incar_settings)
        job.update_metadata({"sample_type": m_data})
        flow = Flow(job, name="Static_flows")
        wf = flow_to_workflow(flow, metadata={"uuid": run_id})
        wfs.append(wf)

    return wfs, run_id


def rerun_crashed_jobs(crashed_jobs, run_id, incar_settings, high_temp_params):
    flow_jobs = []
    for info in crashed_jobs:
        composition = info["composition"]
        strain = info["strain"]
        structure = get_random_packed(composition, target_atoms=100, minAllowDis=1.5, datatype="pymatgen")
        strain_struc = apply_strain_to_structure(structure, [np.eye(3) * (1.0 + strain)])[0].final_structure
        job = md_flow(
            strain_struc,
            name=f"{composition}_{strain}",
            incar_settings=incar_settings,
            temperature=high_temp_params["temperature"],
            steps=high_temp_params["steps"],
        )
        job.update_metadata({"strain": strain, "composition": composition, "sample_type": "high_temp"})
        flow_jobs.append(job)

    flow = Flow(flow_jobs, name="MD_rerun_flows")
    wf = flow_to_workflow(flow, metadata={"uuid": run_id})
    return wf


def train_pace(self, pace_kwargs=None):
    """
    Run the ACE training using pacemaker.

    This function will create a new folder in the ace_fitting directory with the UUID of the run.
    It will then write the input file for pacemaker in this folder, and add a workflow
    to the launchpad to run pacemaker. The trained model will be written to the same folder
    as the input file.
    """
    run_id = str(uuid.uuid4())
    directory = f"{self.wd}/ace_fitting/{run_id}"
    os.makedirs(f"{directory}")
    print(f"Training ACE in {directory}")
    ace_yaml_writer(
        f"{directory}",
        self.database["train"],
        self.database["test"],
        self.atom_types,
        reference_energy=self.reference_energy,
        **pace_kwargs,
    )
    print("Writing input.yaml")
    firetask = ScriptTask.from_str(
        f"cd {directory} ; pacemaker input.yaml ;"
        "pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml"
    )
    wf = Workflow([Firework([firetask], name="training")], metadata={"uuid": run_id}, name="train_ace")
    self.lp.add_wf(wf)
    return wf, directory


def run_lammps(directories, wd, lammps_command, run_id):
    fws = []
    for path in directories:
        command_string = f"cd {path}/ ; {lammps_command} -in {wd}/in.run"
        bash_readable_string = command_string.replace(")", r"\)").replace("(", r"\(")
        firetask = ScriptTask.from_str(bash_readable_string)
        fws.append(Firework(firetask, name=os.path.basename(os.path.normpath(path))))
    wf = Workflow(fws, name="lammps_runs", metadata={"uuid": run_id})
    return wf
