import yaml


def lammps_input_writer(
    pot_dir, atoms, max_temp=5000, min_temp=0.01, cooling_rate=10, sample_rate=100000, seed=1, c_min=1.5, c_max=30
):
    atom_string = " ".join([str(atom) for atom in atoms])

    input = f"""
    #Initialization
    units           metal
    dimension       3
    boundary        p p p
    atom_style      atomic

    read_data structure.dat

    ## in.lammps
    pair_style  pace/extrapolation
    pair_coeff  * * {pot_dir}/output_potential.yaml {pot_dir}/output_potential.asi {atom_string}
    fix pace_gamma all pair 1 pace/extrapolation gamma 1
    compute max_pace_gamma all reduce max f_pace_gamma

    #output
    thermo    100
    thermo_style   custom step temp pe etotal press vol density c_max_pace_gamma
    velocity all create {max_temp} {seed} rot yes dist gaussian

    # dump extrapolative structures if c_max_pace_gamma > 3, skip otherwise, check every 10 steps
    variable dump_skip equal "c_max_pace_gamma < {c_min}"
    dump pace_dump all custom 1 gamma.dump id type x y z f_pace_gamma
    dump_modify pace_dump skip v_dump_skip

    # stop simulation if maximum extrapolation grade exceeds 20
    variable max_pace_gamma equal c_max_pace_gamma
    fix extreme_extrapolation all halt 1 v_max_pace_gamma > {c_max}

    fix 1 all nvt temp {max_temp} {max_temp} 0.1
    run 10000
    unfix 1
    undump pace_dump

    reset_timestep 0

    dump pace_dump all custom 10 gamma.dump id type x y z f_pace_gamma
    dump_modify pace_dump append yes skip v_dump_skip
    dump glass_dump all custom {sample_rate} glass.dump id type x y z

    fix 1 all nvt temp {max_temp} {min_temp} 0.1
    run {int((max_temp-min_temp)*1000 / cooling_rate)}

    unfix 1
    """

    with open("in.ace", "w") as f:
        f.writelines(input)


def ace_yaml_writer(
    wd,
    train_database,
    test_database,
    elements,
    reference_energy="auto",
    cutoff=8.0,
    number_of_functions_per_element=250,
    embeddings={
        "npot": "FinnisSinclairShiftedScaled",
        "fs_parameters": [1, 1, 1, 0.5, 1, 0.75, 1, 0.25, 1, 0.125, 1, 0.375, 1, 0.875, 1, 2],
        "ndensity": 8,
    },
    bonds={
        "radbase": "SBessel",
        "radparameters": [5.25],
        "rcut": 8.0,
        "dcut": 0.01,
        "NameOfCutoffFunction": "cos",
    },
    deltaSplineBins=0.001,
    nradmax_by_orders=[15, 3, 2, 1],
    lmax_by_orders=[0, 4, 2, 0],
    loss={"kappa": 0.05, "L1_coeffs": 1e-8, "L2_coeffs": 1e-8},
    maxiter=2000,
    ladder_steps=5,
    ladder_type="power_order",
    early_stopping_patience=150,
    batch_size=100,
):

    ace_input = {
        "cutoff": cutoff,
        "seed": 42,
        "potential": {
            "deltaSplineBins": deltaSplineBins,
            "elements": elements,
            "embeddings": {"ALL": embeddings},
            "bonds": {"ALL": bonds},
            "functions": {
                "number_of_functions_per_element": number_of_functions_per_element,
                "UNARY": {"nradmax_by_orders": nradmax_by_orders, "lmax_by_orders": lmax_by_orders},
                "BINARY": {"nradmax_by_orders": nradmax_by_orders, "lmax_by_orders": lmax_by_orders},
                "TERNARY": {"nradmax_by_orders": nradmax_by_orders, "lmax_by_orders": lmax_by_orders},
                "QUATERNARY": {"nradmax_by_orders": nradmax_by_orders, "lmax_by_orders": lmax_by_orders},
            },
        },
        "data": {"filename": train_database, "test_filename": test_database, "reference_energy": reference_energy},
        "fit": {
            "loss": loss,
            "optimizer": "BFGS",
            "repulsion": "auto",
            "maxiter": maxiter,
            "ladder_steps": ladder_steps,
            "ladder_type": ladder_type,
            "min_relative_train_loss_per_iter": 5e-5,
            "min_relative_test_loss_per_iter": 1e-5,
            "early_stopping_patience": early_stopping_patience,
        },
        "backend": {
            "evaluator": "tensorpot",
            "batch_size": batch_size,
            "display_step": 100,
            "gpu_config": {"gpu_ind": -1, "mem_limit": 0},
        },
    }
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(f"{wd}/input.yaml", "w") as f:
        yaml.dump(ace_input, f)
