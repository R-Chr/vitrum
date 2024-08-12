def lammps_input_writer(
    wd, atoms, max_temp=5000, min_temp=300, cooling_rate=10, sample_rate=10000, seed=1, c_min=3, c_max=20
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
    pair_coeff  * * {wd}/output_potential.yaml {wd}/output_potential.asi {atom_string}
    fix pace_gamma all pair 1 pace/extrapolation gamma 1
    compute max_pace_gamma all reduce max f_pace_gamma

    #output
    thermo    100
    thermo_style   custom step temp pe etotal press vol density c_max_pace_gamma
    velocity all create {max_temp} {seed} rot yes dist gaussian

    # dump extrapolative structures if c_max_pace_gamma > 3, skip otherwise, check every 10 steps
    variable dump_skip equal "c_max_pace_gamma < {c_min}"
    dump pace_dump all custom 10 gamma.dump id type x y z f_pace_gamma
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

    fix 1 all nvt temp {max_temp} {300} 0.1
    run {int((max_temp-min_temp)*1000 / cooling_rate)}

    unfix 1
    """

    with open("in.ace", "w") as f:
        f.writelines(input)
