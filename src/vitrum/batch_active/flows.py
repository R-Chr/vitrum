from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.md import MDMaker
from pymatgen.io.vasp import Kpoints
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.sets.core import MDSetGenerator


def static_flow(structure, name=False, incar_settings={}, kpoint=False, potcar_functional="PBE_54"):
    if not name:
        name = structure.reduced_formula
    num_atoms = len(structure)
    incar_set = {
        "EDIFF": (10**-5) * num_atoms,
        "ENAUG": None,
        "EDIFFG": None,
        "ALGO": "Normal",
        "ENCUT": 520,
        "ISMEAR": 0,
        "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
        "LREAL": "Auto",
        "LAECHG": False,
        "LASPH": True,
        "LCHARG": False,
        "GGA": None,  # Just let VASP decide based on POTCAR - the default PE
        "LPLANE": False,  # LPLANE is recommended to be False on Cray machines
        "LDAUPRINT": 0,
        "ISIF": 2,
        "SIGMA": 0.05,
        "LVTOT": None,
        "LMIXTAU": None,
        "NELM": 200,
        "PREC": "Normal",
        "NCORE": 16,
        "NSIM": 32,
    }
    incar_set.update(incar_settings)
    if not kpoint:
        kpoint = Kpoints()  # Gamma centered, 1x1x1 KPOINTS with no shift

    run_vasp_kwargs = {"job_type": "direct"}

    static_maker = StaticMaker(
        name=name,
        input_set_generator=StaticSetGenerator(
            user_incar_settings=incar_set, user_kpoints_settings=kpoint, user_potcar_functional=potcar_functional
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
    incar_settings={},
    kpoint=False,
    potcar_functional="PBE_54",
):
    if not name:
        name = structure.reduced_formula
    num_atoms = len(structure)

    incar_set = {
        "EDIFF": (10**-5) * num_atoms,
        "ENAUG": None,
        "EDIFFG": None,
        "ALGO": "Normal",
        "ENCUT": 520,
        "ISMEAR": 0,
        "ISPIN": 1,  # Do not consider magnetism in AIMD simulations
        "LREAL": "Auto",
        "LAECHG": False,
        "LASPH": True,
        "LCHARG": False,
        "GGA": None,  # Just let VASP decide based on POTCAR - the default PE
        "LPLANE": False,  # LPLANE is recommended to be False on Cray machines
        "LDAUPRINT": 0,
        "ISIF": 2,
        "SIGMA": 0.05,
        "LVTOT": None,
        "LMIXTAU": None,
        "NELM": 200,
        "PREC": "Normal",
        "NCORE": 16,
        "NSIM": 32,
    }

    incar_set.update(incar_settings)

    if not kpoint:
        kpoint = Kpoints()  # Gamma centered, 1x1x1 KPOINTS with no shift

    run_vasp_kwargs = {"job_type": "direct"}

    aimd_maker = MDMaker(
        name=name,
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=temperature,
            nsteps=steps,
            time_step=timestep,
            # adapted from MPMorph settings
            user_incar_settings=incar_set,
            user_kpoints_settings=kpoint,
            user_potcar_functional=potcar_functional,
        ),
        run_vasp_kwargs=run_vasp_kwargs,
    )

    return aimd_maker.make(structure)
