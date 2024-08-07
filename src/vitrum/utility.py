from ase import Atoms
from pymatgen.ext.matproj import MPRester
import numpy as np
from pymatgen.core import Composition, Structure
from collections import deque
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
import warnings


def get_random_packed(
    composition, density=None, target_atoms=100, minAllowDis=1.7, mp_api_key=None, datatype="ase", seed=None
):
    """
    Generate a random packed structure based on the given composition.

    Args:
        composition (str, dict or pymatgen.core.Composition): The composition of the structure.
        density (float, optional): The target density of the structure (in g/cm^3). If not provided, the volume per atom
                                   is estimated using the Materials Project API.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        minAllowDis (float, optional): The minimum allowed distance between atoms (in angstroms). Defaults to 1.7.
        mp_api_key (str, optional): The API key for the Materials Project. Required if density is not provided.
        datatype (str, optional): The type of data to return. Can be "ase" for ASE format or "pymatgen"
                                  for pymatgen format. Defaults to "ase".
        seed (int, optional): The seed for random number generation. Defaults to 0.

    Returns:
        ase.Atoms or pymatgen.core.Structure: The generated random packed structure.

    Raises:
        ValueError: If density is not provided and mp_api_key is not provided.

    """
    if isinstance(composition, str):
        composition = Composition(composition)
    elif isinstance(composition, dict):
        comp_string = "".join(mol * (int(composition[mol] * 10)) for mol in composition)
        composition = Composition(comp_string)
    formula, factor = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(target_atoms / integer_composition.num_atoms)
    structure = {}
    for el in full_cell_composition:
        structure[str(el)] = int(full_cell_composition.element_composition.get(el))

    if not density:
        if not mp_api_key:
            density = 2.5
            warnings.warn("No density or MP API key provided, setting density to 2.5 g/cm3")
        else:
            mpr = MPRester(mp_api_key)
            _entries = mpr.get_entries_in_chemsys([str(el) for el in composition.elements], inc_structure=True)
            entries = []
            for entry in _entries:
                if set(entry.structure.composition.elements) == set(composition.elements):
                    entries.append(entry)
                if len(entry.structure.composition.elements) >= 2:
                    entries.append(entry)
            vols = [entry.structure.volume / entry.structure.num_sites for entry in entries]
            vol_per_atom = np.mean(vols)
            cell_len = (vol_per_atom * full_cell_composition.num_atoms) ** (1 / 3)
    else:
        mass = np.sum([Atoms(f"{i}").get_masses()[0] * structure[i] for i in structure])
        cell_vol = (mass / (6.0221 * (10**23))) / density
        cell_len = cell_vol ** (1 / 3) * (10**8)

    cell = np.array([cell_len, cell_len, cell_len])
    pos = np.array([[0, 0, 0]])
    if seed:
        np.random.seed(seed)

    escape_counter = 0
    while len(pos) < full_cell_composition.num_atoms:
        xyz_pos = np.random.rand(1, 3) * cell
        delta = np.abs(xyz_pos - pos)
        delta = np.where(delta > 0.5 * cell_len, np.abs(delta - cell_len), delta)
        if np.all(delta > minAllowDis):
            pos = np.append(pos, xyz_pos, axis=0)
            escape_counter = 0
        else:
            delta = np.sqrt((delta**2).sum(axis=1))
            if np.all(delta > minAllowDis):
                pos = np.append(pos, xyz_pos, axis=0)
                escape_counter = 0
            else:
                escape_counter += 1
                if escape_counter > 1000:
                    raise ValueError(
                        "Error: Cannot find suitable positions for atoms, lower minAllowDis or decreasing the density"
                    )

    formula = sum([[i] * structure[i] for i in structure], [])
    if datatype == "ase":
        data = Atoms(formula, positions=pos, cell=cell, pbc=True)
    elif datatype == "pymatgen":
        lattice_vectors = [[0, 0, 0] for _ in range(3)]
        for i in range(3):
            lattice_vectors[i][i] = cell_len
        data = Structure(lattice_vectors, formula, pos, coords_are_cartesian=True)
    return data


def get_LAMMPS_dump_timesteps(filename: str):
    """
    Retrieves the timesteps from a LAMMPS dump file.

    Parameters:
        filename (str): The path to the LAMMPS dump file.

    Returns:
        List[int]: A list of timesteps extracted from the file.
    """
    with open(filename, encoding="utf-8") as f:
        timesteps = []
        lines = deque(f.readlines())
        line = lines.popleft()
        while len(lines) > 0:
            if "ITEM: TIMESTEP" in line:
                line = lines.popleft()
                timesteps.append(int(line))
            else:
                line = lines.popleft()
    return timesteps


def apply_strain_to_structure(structure, deformations: list) -> list:
    """
    Apply strain(s) to input structure and return transformation(s) as list.

    Parameters
    ----------
    structure: .Structure
        Input structure to apply strain to
    deformations: list[.Deformation]
        A list of deformations to apply **independently** to the input
        structure, in anticipation of performing an EOS fit.
        Deformations should be of the form of a 3x3 matrix, e.g.,::

        [[1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]]

        or::

        ((1.2, 0., 0.), (0., 1.2, 0.), (0., 0., 1.2))

    Returns
    -------
    list
        A list of .TransformedStructure objects corresponding to the
        list of input deformations.
    """
    transformations = []
    for deformation in deformations:
        # deform the structure
        ts = TransformedStructure(
            structure,
            transformations=[DeformStructureTransformation(deformation=deformation)],
        )
        transformations += [ts]
    return transformations
