import os

import numpy as np
from ase.io.lammpsdata import write_lammps_data
from pymatgen.io.ase import AseAtomsAdaptor

from vitrum.packing import apply_strain_to_structure
from vitrum.structure_gen import GlassGenerator


def gen_even_structures(
    units,
    spacing: int = 10,
    datatype: str = "pymatgen",
    target_atoms: int = 100,
    minAllowDis: float = 1.7,
    **kwargs,
) -> list:
    """
    Generate a list of structures with compositions spaced evenly between 0 and 100
    percent of each unit.

    Thin wrapper around ``GlassGenerator(units=...).sample("grid")`` followed by
    packing. New code should use :class:`vitrum.structure_gen.GlassGenerator` directly.

    Parameters:
        units: list
            Composition units (e.g. oxide formulas) to grid over.
        spacing: int, optional
            Spacing between each composition point, by default 10
        datatype: str, optional
            Type of structure to return, either "pymatgen" or "ase", by default "pymatgen"
        **kwargs: dict, optional
            Additional keyword arguments to pass to get_random_packed

    Returns:
        structures: list
            List of structures with evenly spaced compositions
    """
    generator = GlassGenerator(units=units)
    comps = generator.sample("grid", spacing=spacing)
    return comps.get_structures(
        target_atoms=target_atoms,
        datatype=datatype,
        min_distance=minAllowDis,
        **kwargs,
    )


def gen_strained_structures(structure, max_strain=0.2, num_strains=3):
    """
    Generate a list of structures with linear strains applied to the given structure.

    Parameters:
        structure : pymatgen.Structure
            The structure to apply the strain to
        max_strain : float, optional
            The maximum strain to apply, by default 0.2
        num_strains : int, optional
            The number of strains to apply, by default 3

    Returns:
        struc: list
            List of structures with linear strains applied
        linear_strain: list
            List of the linear strain values applied
    """
    linear_strain = np.linspace(-max_strain, max_strain, num_strains)
    strain_matrices = [np.eye(3) * (1.0 + eps) for eps in linear_strain]
    strained_structures = apply_strain_to_structure(structure, strain_matrices)
    struc = [strained_structures[index].final_structure for index in range(len(strain_matrices))]
    return struc, linear_strain


def gen_lammps_structures(structures, strain_params, specorder, path):
    paths = []
    for index, structure in enumerate(structures):
        name = structure.reduced_formula
        strained_structures, linear_strain = gen_strained_structures(
            structure, strain_params["max_strain"], strain_params["num_strains"]
        )
        for strain, strain_struc in zip(linear_strain, strained_structures):
            strain_struc = AseAtomsAdaptor().get_atoms(strain_struc)
            new_dir = f"{path}/{name}_{strain}_{index}"
            os.makedirs(new_dir)
            write_lammps_data(
                f"{new_dir}/structure.dat",
                strain_struc,
                masses=True,
                specorder=specorder,
            )
            paths.append(new_dir)
    return paths
