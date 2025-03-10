from ase import Atoms
from mp_api.client import MPRester
import numpy as np
from pymatgen.core import Composition, Structure
from collections import deque
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
import warnings
from scipy.signal import argrelextrema
from ase.data import covalent_radii, atomic_numbers


def get_random_packed(
    composition,
    density=None,
    target_atoms=100,
    minAllowDis=1.7,
    mp_api_key=None,
    datatype="ase",
    seed=None,
    side_ratios=[1, 1, 1],
):
    """
    Generate a random packed structure based on the given composition.

    Parameters:
        composition (str, dict or pymatgen.core.Composition): The composition of the structure.
        density (float, optional): The target density of the structure (in g/cm^3). If not provided, the volume per atom
                                   is estimated using the Materials Project API.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        minAllowDis (float, optional): The minimum allowed distance between atoms (in angstroms). Defaults to 1.7.
        mp_api_key (str, optional): The API key for the Materials Project. Required if density is not provided.
        datatype (str, optional): The type of data to return. Can be "ase" for ASE format or "pymatgen"
                                  for pymatgen format. Defaults to "ase".
        seed (int, optional): The seed for random number generation. Defaults to 0.
        side_ratios (list, optional): The side ratios for the lattice. Defaults to [1, 1, 1].

    Returns:
        data (ase.Atoms or pymatgen.core.Structure): The generated random packed structure.
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
    np.random.seed(seed)

    if not mp_api_key and not density:
        density = 2.5
        warnings.warn("No MP API key provided, setting density to 2.5 g/cm3")

    if mp_api_key:
        mpr = MPRester(mp_api_key, mute_progress_bars=True)
        comp_entries = mpr.get_entries(composition.reduced_formula)
        if len(comp_entries) > 0:
            vols = np.min([entry.structure.volume / entry.structure.num_sites for entry in comp_entries])
        else:
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], additional_criteria={"is_stable": True}
            )
            entries = []
            for entry in _entries:
                if set(entry.structure.composition.elements) == set(composition.elements):
                    entries.append(entry)
                if len(entry.structure.composition.elements) >= 2:
                    entries.append(entry)
            vols = [entry.structure.volume / entry.structure.num_sites for entry in entries]

        vol_per_atom = np.mean(vols)
        cell_vol = vol_per_atom * full_cell_composition.num_atoms

    if density:
        mass = np.sum([Atoms(f"{i}").get_masses()[0] * structure[i] for i in structure])
        cell_vol = ((mass / (6.0221 * (10**23))) / density) * (10**24)

    if covalent_radii:
        all_radii = np.hstack(
            [np.repeat(covalent_radii[atomic_numbers[key]], structure[key]) for key in structure.keys()]
        )
        cell_vol = np.sum((4 / 3 * np.pi * all_radii**3)) * 3

    k = (cell_vol / (side_ratios[0] * side_ratios[1] * side_ratios[2])) ** (1 / 3)
    cell = np.array([side_ratios[0] * k, side_ratios[1] * k, side_ratios[2] * k])

    tries = 0
    while tries < 10:
        pos = np.array([[0, 0, 0]])
        escape_counter = 0
        while len(pos) < full_cell_composition.num_atoms:
            xyz_pos = np.random.rand(1, 3) * cell
            delta = np.abs(xyz_pos - pos)
            delta = np.where(delta > 0.5 * cell, np.abs(delta - cell), delta)
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
                        break
        if len(pos) == full_cell_composition.num_atoms:
            break
    if tries == 10:
        raise ValueError("Error: Cannot find suitable positions for atoms, lower minAllowDis or decrease the density")

    formula = sum([[i] * structure[i] for i in structure], [])
    if datatype == "ase":
        data = Atoms(formula, positions=pos, cell=cell, pbc=True)
    elif datatype == "pymatgen":
        lattice_vectors = [[0, 0, 0] for _ in range(3)]
        for i in range(3):
            lattice_vectors[i][i] = cell[i]
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
        if len(lines) == 0:
            return timesteps
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

    Parameters:
        structure (.Structure): Input structure to apply strain to
        deformations (list[.Deformation]): A list of deformations to apply independently to the input
            structure, in anticipation of performing an EOS fit.
            Deformations should be of the form of a 3x3 matrix, e.g.,

            [[1.2, 0., 0.], [0., 1.2, 0.], [0., 0., 1.2]]

            or

            ((1.2, 0., 0.), (0., 1.2, 0.), (0., 0., 1.2))

    Returns:
        list: A list of .TransformedStructure objects corresponding to the
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


def find_min_after_peak(self, padf):
    mins = argrelextrema(padf, np.less_equal, order=4)[0]
    second_min = [i for ind, i in enumerate(mins) if i != ind][0]
    return second_min


def correct_atom_types(atoms_list, atom_to_type_map):
    """
    Correct the atom types in a list of Atoms objects.

    Parameters:
        atoms_list (list of Atoms objects): The list of Atoms objects to correct.
        atom_to_type_map (dict): A dictionary mapping atomic numbers to atom types.

    Returns:
        atoms_list (list of Atoms objects): The corrected list of Atoms objects.
    """

    for atoms in atoms_list:
        corr_symbols = [atom_to_type_map[i] for i in atoms.get_atomic_numbers()]
        atoms.set_chemical_symbols(corr_symbols)
    return atoms_list


def get_high_low_displacement_index(initial_state, current_state, target_atom, percentage=0.25):
    """
    Calculates the indices of the atoms with the high and low displacements between an initial and current state.

    Parameters:
        initial_state (Atoms): The initial state of the system.
        current_state (Atoms): The current state of the system.
        target_atom (str or int): The chemical symbol or atomic number of the target atom.
        percentage (float, optional): The percentage of the highest and lowest displacements to consider. Defaults to 0.25.

    Returns:
        list: A list of two elements, where the first element is the index of the atoms with the highest displacements
            and the second element is the index of the atoms with the lowest displacements.
    """
    index = np.where(initial_state.get_chemical_symbols() == target_atom)[0]
    initial_positions = initial_state.get_positions()[index]
    current_positions = current_state.get_positions()[index]
    displacements = initial_positions - current_positions
    displacements = np.sum(displacements**2, axis=1)
    ind = np.argsort(displacements)
    low_ind = index[ind[: int(len(ind) * percentage)]]
    high_ind = index[ind[int(len(ind) * percentage) :]]
    return [low_ind, high_ind]


def unwrap_trajectory(atoms_list):
    """
    Unwraps a list of Atoms objects to remove periodic boundary crossings.

    Parameters:
        atoms_list (list of Atoms objects): The list of Atoms objects to unwrap.

    Returns:
        unwrapped_atoms_list (list of Atoms objects): The unwrapped list of Atoms objects.
    """

    if not atoms_list or len(atoms_list) == 0:
        raise ValueError("The input atoms_list must be a non-empty list of ASE Atom objects.")
    cell = np.diagonal(atoms_list[0].get_cell())
    n_atoms = len(atoms_list[0])
    unwrapped_atoms_list = [atoms.copy() for atoms in atoms_list]

    crossings = np.zeros((n_atoms, 3))
    previous_positions = unwrapped_atoms_list[0].get_positions()
    for atoms in unwrapped_atoms_list:
        current_positions = atoms.get_positions()
        difference = current_positions - previous_positions
        crossings = crossings + np.floor_divide(difference + 0.5 * cell, cell).astype(int)
        previous_positions = current_positions
        new_positions = current_positions - cell * crossings
        atoms.set_positions(new_positions)
    return unwrapped_atoms_list
