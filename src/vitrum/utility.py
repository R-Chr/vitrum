from ase import Atoms
import numpy as np
from pymatgen.core import Composition, Structure
from collections import deque
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from scipy.signal import argrelextrema
from ase.data import covalent_radii, atomic_numbers
from atomate2.common.jobs.mpmorph import get_average_volume_from_mp, get_average_volume_from_db_cached


def get_volume(
    composition: Composition | str,
    structure: dict,
    vol_per_atom_source: float | str = "mp",
    db_kwargs: dict | None = None,
    density: float | None = None,
):

    struct_db = vol_per_atom_source.lower() if isinstance(vol_per_atom_source, str) else None
    db_kwargs = db_kwargs or ({"use_cached": True} if struct_db == "mp" else {})
    cell_vol = None

    if density:
        struct_db == "density"

    if isinstance(vol_per_atom_source, float | int):
        vol_per_atom = vol_per_atom_source

    elif struct_db == "mp":
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)

    elif struct_db == "icsd":
        vol_per_atom = get_average_volume_from_db_cached(composition, db_name="icsd", **db_kwargs)

    elif struct_db == "density":
        if not density:
            raise ValueError("Must specify a valid density")
        mass = np.sum([Atoms(f"{i}").get_masses()[0] * structure[i] for i in structure])
        cell_vol = ((mass / (6.0221 * (10**23))) / density) * (10**24)

    elif struct_db == "covalent_radius":
        all_radii = np.hstack(
            [np.repeat(covalent_radii[atomic_numbers[key]], structure[key]) for key in structure.keys()]
        )
        cell_vol = np.sum((4 / 3 * np.pi * all_radii**3)) * 3

    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source}.")

    if not cell_vol:
        cell_vol = vol_per_atom * sum(structure.values())

    return cell_vol


def get_random_packed(
    composition: str | dict | Composition,
    target_atoms: int = 100,
    minAllowDis: float = 1.7,
    vol_per_atom_source: float | str = "mp",
    datatype: str = "ase",
    db_kwargs: dict | None = None,
    density: float | None = None,
    seed: int | None = None,
    side_ratios: list = [1, 1, 1],
):
    """
    Generate a random packed structure based on the given composition.

    Parameters:
        composition (str, dict or pymatgen.core.Composition): The composition of the structure.
        density (float, optional): The target density of the structure (in g/cm^3). If not provided, the volume per atom
                                   is estimated using the Materials Project API.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        minAllowDis (float, optional): The minimum allowed distance between atoms (in angstroms). Defaults to 1.7.
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
    cell_vol = get_volume(composition, structure, vol_per_atom_source, db_kwargs, density)

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


def pdf(dist_list, volume, rrange=10, nbin=100):
    """
    Calculate the pair distribution function (PDF) of a list of distances.

    Parameters:
        dist_list (np.ndarray): A 2D numpy array of distances.
        volume (float): The volume of the system.
        rrange (float, optional): The range of the PDF. Defaults to 10.
        nbin (int, optional): The number of bins. Defaults to 100.

    Returns:
        xval (np.ndarray): The x values of the PDF.
        pdf (np.ndarray): The PDF values.
    """

    edges = np.linspace(0, rrange, nbin + 1)
    xval = edges[1:] - 0.5 * (rrange / nbin)
    volbin = []
    for i in range(nbin):
        vol = ((4 / 3) * np.pi * (edges[i + 1]) ** 3) - ((4 / 3) * np.pi * (edges[i]) ** 3)
        volbin.append(vol)

    h, bin_edges = np.histogram(dist_list, bins=nbin, range=(0, rrange))
    h[0] = 0
    pdf = (h / volbin) / (dist_list.shape[0] * dist_list.shape[1] / volume)
    return xval, pdf


def get_dist(list, cell):
    """
    Calculate the distance between atoms in a box with PBC and 90 degree angles."

    Parameters:
        list (np.ndarray): A 2D numpy array of atomic positions.
        cell (np.ndarray): The cell dimensions of the system

    Returns:
        i_i (np.ndarray): The interatomic distances.
    """
    dim = [cell[0], cell[1], cell[2]]
    x_dif = np.abs(list[:, 0][np.newaxis, :] - list[:, 0][:, np.newaxis])
    y_dif = np.abs(list[:, 1][np.newaxis, :] - list[:, 1][:, np.newaxis])
    z_dif = np.abs(list[:, 2][np.newaxis, :] - list[:, 2][:, np.newaxis])
    x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
    y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
    z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
    i_i = np.sqrt(x_dif**2 + y_dif**2 + z_dif**2)
    return i_i
