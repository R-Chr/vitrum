from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.md import MDMaker
from pymatgen.io.vasp import Kpoints
from jobflow import Flow
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.core import MDSetGenerator
import numpy as np
from vitrum.utility import apply_strain_to_structure


def static_flow(structure, name=False, incar_settings=False, kpoint=False, potcar_functional="PBE_64"):
    if not name:
        name = structure.reduced_formula

    if not incar_settings:
        num_atoms = len(structure)
        incar_settings = {
            "EDIFF": (10**-5) * num_atoms,
            "ENAUG": None,
            "EDIFFG": None,
            "ENCUT": 520,
            "ISMEAR": 0,
            "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
            "LREAL": "Auto",  # Peform calculation in real space for AIMD due to large unit cell size
            "LAECHG": False,  # Don't need AECCAR for AIMD
            "LCHARG": False,
            "GGA": None,  # Just let VASP decide based on POTCAR - the default PE
            "LPLANE": False,  # LPLANE is recommended to be False on Cray machines
            "LDAUPRINT": 0,
            "ISIF": 1,
            "SIGMA": 0.05,
            "LVTOT": None,
            "LMIXTAU": None,
            "NELM": 500,
            "PREC": "Normal",
        }

    if not kpoint:
        kpoint = Kpoints(
            comment="Gamma only",
            num_kpts=1,
            kpts=[[0, 0, 0]],
            kpts_weights=[1.0],
        )

    run_vasp_kwargs = {"job_type": "direct"}

    static_maker = StaticMaker(
        name=name,
        input_set_generator=StaticSetGenerator(
            user_incar_settings=incar_settings, user_kpoints_settings=kpoint, user_potcar_functional=potcar_functional
        ),
        run_vasp_kwargs=run_vasp_kwargs,
    )

    return static_maker.make(structure)


def md_flow(
    structure,
    temperature=5000,
    steps=100,
    name=False,
    timestep=1,
    incar_settings=False,
    kpoint=False,
    potcar_functional="PBE_64",
):
    if not name:
        name = structure.reduced_formula

    if not incar_settings:
        num_atoms = len(structure)
        incar_settings = {
            "EDIFF": (10**-5) * num_atoms,
            "ENAUG": None,
            "EDIFFG": None,
            "ENCUT": 520,
            "ISMEAR": 0,
            "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
            "LREAL": "Auto",  # Peform calculation in real space for AIMD due to large unit cell size
            "LAECHG": False,  # Don't need AECCAR for AIMD
            "LCHARG": False,
            "GGA": None,  # Just let VASP decide based on POTCAR - the default PE
            "LPLANE": False,  # LPLANE is recommended to be False on Cray machines
            "LDAUPRINT": 0,
            "ISIF": 1,
            "SIGMA": 0.05,
            "LVTOT": None,
            "LMIXTAU": None,
            "NELM": 500,
            "PREC": "Normal",
        }

    if not kpoint:
        kpoint = Kpoints(
            comment="Gamma only",
            num_kpts=1,
            kpts=[[0, 0, 0]],
            kpts_weights=[1.0],
        )

    aimd_maker = MDMaker(
        name=name,
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=temperature,
            nsteps=steps,
            time_step=timestep,
            # adapted from MPMorph settings
            user_incar_settings=incar_settings,
            user_kpoints_settings=kpoint,
            user_potcar_functional=potcar_functional,
        ),
    )

    return aimd_maker.make(structure)


def strained_flows(structure, name=False, max_strain=0.2, num_strains=3, metadata=None):
    if not name:
        name = structure.reduced_formula

    linear_strain = np.linspace(-max_strain, max_strain, num_strains)
    strain_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
    strained_structures = apply_strain_to_structure(structure, strain_matrices)
    flow_jobs = [
        md_flow(strain_struc, name=strain) for strain, strain_struc in zip(linear_strain, strained_structures)
    ]
    flow = Flow(flow_jobs, name=name)
    if not metadata:
        flow.update_metadata({"metadata": f"{metadata}"})
    return flow
