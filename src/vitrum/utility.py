from ase import Atoms
import numpy as np
from pymatgen.core import Composition, Structure
from collections import deque
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from scipy.signal import argrelextrema
from ase.data import covalent_radii, atomic_numbers
from atomate2.common.jobs.mpmorph import get_average_volume_from_mp, get_average_volume_from_db_cached, get_average_volume_from_mp_api
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.optimize import FIRE
from itertools import product
from ase.neighborlist import NeighborList
from ase.symbols import symbols2numbers
from mp_api.client import MPRester, MPRestError
from pymatgen.analysis.phase_diagram import PhaseDiagram
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit, prange

def get_volume(
    composition: Composition | str,
    structure: dict,
    vol_per_atom_source: float | str = "mp",
    db_kwargs: dict | None = None,
    density: float | None = None,
    MP_API_KEY: str | None = None,
):

    
    struct_db = vol_per_atom_source.lower() if isinstance(vol_per_atom_source, str) else None
    db_kwargs = db_kwargs or ({"use_cached": True} if struct_db == "mp" else {})
    cell_vol = None

    if density:
        if not isinstance(density, (float, int)):
            raise ValueError("Density must be a float or int.")
        
        struct_db = "density"

    if isinstance(vol_per_atom_source, float | int):
        vol_per_atom = vol_per_atom_source

    elif struct_db == "mp":
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)

    elif struct_db == "icsd":
        vol_per_atom = get_average_volume_from_db_cached(composition, db_name="icsd", **db_kwargs)

    elif struct_db == "density":
        mass = np.sum([Atoms(f"{i}").get_masses()[0] * structure[i] for i in structure])
        cell_vol = ((mass / (6.0221 * (10**23))) / density) * (10**24)

    elif struct_db == "covalent_radius":
        all_radii = np.hstack(
            [np.repeat(covalent_radii[atomic_numbers[key]], structure[key]) for key in structure.keys()]
        )
        cell_vol = np.sum((4 / 3 * np.pi * all_radii**3)) * 3

    elif struct_db == "convex_hull":
        try:
            vol_per_atom = get_average_volume_convex_hull(composition, MP_API_KEY=MP_API_KEY)
        except MPRestError as e:
            raise ValueError(f"Could not retrieve volume from convex hull. Check your MP_API_KEY. Error: {e}")

    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source}.")

    if not cell_vol:
        cell_vol = vol_per_atom * sum(structure.values())

    return cell_vol

def get_average_volume_convex_hull(composition, MP_API_KEY=None):
    with MPRester(api_key=MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements=[str(el) for el in composition.elements],
            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
        )
    pd = PhaseDiagram(entries)
    decomp = pd.get_decomposition(composition)
    volume = sum([d.structure.volume / d.composition.num_atoms * decomp[d] for d in decomp])
    return volume

def get_random_packed(
    composition: str | dict | Composition,
    target_atoms: int = 100,
    min_distance: float | None = None,
    radii_scaling: float = 1.0,
    volume_scaling: float = 1.0,
    vol_per_atom_source: float | str = "mp",
    datatype: str = "ase",
    db_kwargs: dict | None = None,
    density: float | None = None,
    seed: int | None = None,
    side_ratios: list = [1, 1, 1],
    **kwargs,
):
    """
    Generate a random packed structure based on the given composition.

    Parameters:
        composition (str, dict or pymatgen.core.Composition): The composition of the structure.
        density (float, optional): The target density of the structure (in g/cm^3). If not provided, the volume per atom
                                   is estimated using the Materials Project API.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        radii_scaling (float, optional): Scaling factor for the covalent radii of the atoms. Defaults to 1.0.
        volume_scaling (float, optional): Scaling factor for the volume of the structure. Defaults to 1.0.
        vol_per_atom_source (float or str, optional): The source for the volume per atom. Can be a float value or one of the following strings:
                                                    "mp" (Materials Project), "icsd" (Inorganic Crystal Structure Database),
                                                     "density" (use provided density), "covalent_radius" (estimate from covalent radii),
                                                     or "convex_hull" (estimate from convex hull). Defaults to "mp".
        datatype (str, optional): The type of data to return. Can be "ase" for ASE format or "pymatgen"
                                  for pymatgen format. Defaults to "ase".
        db_kwargs (dict, optional): Additional keyword arguments for database access. Defaults to None.
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
    elements, factor = composition.get_integer_formula_and_factor()
    integer_composition = Composition(elements)
    full_cell_composition = integer_composition * np.ceil(target_atoms / integer_composition.num_atoms)

    structure = {}
    for el in full_cell_composition:
        structure[str(el)] = int(full_cell_composition.element_composition.get(el))
    elements = sum([[i] * structure[i] for i in structure], [])
    np.random.seed(seed)

    cell_vol = get_volume(composition, structure, vol_per_atom_source, db_kwargs, density, **kwargs)

    cell_vol *= volume_scaling
    k = (cell_vol / (side_ratios[0] * side_ratios[1] * side_ratios[2])) ** (1 / 3)
    cell = np.array([side_ratios[0] * k, side_ratios[1] * k, side_ratios[2] * k])
    cell = np.diag(cell)

    radii = covalent_radii[symbols2numbers(elements)] * radii_scaling
    
    if min_distance:
        radii = np.maximum(radii, min_distance / 2)

    nat = len(elements)
    
    pos = np.random.rand(len(elements), 3) @ cell
    ats = Atoms(elements, cell=cell, pbc=True, positions=pos)
    skin = 0.0
    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=skin)

    for it in range(500):
        nl.update(ats)
        pos = ats.get_positions()
        dx = np.zeros(pos.shape)
        dsum = 0
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            rs = pos[indices, :] + offsets @ cell - pos[i, :]
            # ds is the overlap
            ds = np.linalg.norm(rs, axis=1) - (radii[indices] + radii[i])
            if np.any(ds > 1.e-10):
                print('Assertion failed: ds <= 0')
                print(np.max(ds))
                quit()
            # sum overlaps 
            dsum += np.sum(ds) 
            ds -= skin
            # move atoms away from each other by overlap amount
            dx[i,:] = np.sum(rs / np.linalg.norm(rs, axis=1)[:, None] * ds[:, None], axis=0)
        # print(it, dsum, np.linalg.norm(dx))
        ats.set_positions(pos + dx)
        if dsum >= -1.e-5:
            break
    else:
        raise RuntimeError('Cell packing not converged')

    ats.wrap()
    if datatype == "pymatgen":
        structure = Structure(
            lattice=ats.get_cell(),
            species=ats.get_chemical_symbols(),
            coords=ats.get_scaled_positions(),
            to_unit_cell=True,
            coords_are_cartesian=False,
        )
    else:
        structure = ats
    return structure



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
    #Check if atoms_list is a list of Atoms objects
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    for atoms in atoms_list:
        corr_symbols = [atom_to_type_map[i] for i in atoms.get_atomic_numbers()]
        atoms.set_chemical_symbols(corr_symbols)


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
    xval = (edges[1:] + edges[:-1]) / 2
    volbin = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    h, bin_edges = np.histogram(dist_list, bins=nbin, range=(0, rrange))
    h[0] = 0
    pdf = (h / volbin) / (dist_list.size / volume)
    return xval, pdf

@njit(parallel=True)
def get_dist_numba(pos, cell):
    n = pos.shape[0]
    # Initialize the output matrix
    dist_matrix = np.zeros((n, n))
    
    # Extract cell dimensions for faster access
    lx, ly, lz = cell[0], cell[1], cell[2]
    half_lx, half_ly, half_lz = lx / 2.0, ly / 2.0, lz / 2.0

    for i in prange(n):
        for j in range(i + 1, n):  # Only calculate the upper triangle
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            dz = abs(pos[i, 2] - pos[j, 2])

            # Apply Periodic Boundary Conditions (Minimum Image Convention)
            if dx > half_lx: dx -= lx
            if dy > half_ly: dy -= ly
            if dz > half_lz: dz -= lz

            d = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Fill both symmetric entries
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    return dist_matrix

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


def r_chi(function_1, function_2, x_min=0, x_max=np.inf, steps=100):
    """
    Calculate the Wright coefficient (https://doi.org/10.1016/0022-3093(93)90232-M) between two functions

    Parameters:
        function_1 (dict): Dictionary with keys 'x' and 'y' representing the first function, usually from simulations
        function_2 (dict): Dictionary with keys 'x' and 'y' representing the second function, usually from experimental meassurements.

    Returns:
        rchi (float): Wright coefficient, a measure of similarity between the two functions.
    """
    # Determine the overlapping x-range
    min_x_val = np.max([np.min(function_1["x"]), np.min(function_2["x"]), x_min])
    max_x_val = np.min([np.max(function_1["x"]), np.max(function_2["x"]), x_max])

    if min_x_val >= max_x_val:
        raise ValueError("No overlapping x-range between the two functions.")

    # Create common x-axis over the overlap region
    common_x = np.linspace(min_x_val, max_x_val, steps)

    # Interpolate both functions onto the common x-axis
    interp_f1 = interp1d(function_1["x"], function_1["y"], kind="linear", bounds_error=False, fill_value=0)
    interp_f2 = interp1d(function_2["x"], function_2["y"], kind="linear", bounds_error=False, fill_value=0)

    y1 = interp_f1(common_x)
    y2 = interp_f2(common_x)

    # Calculate the r-chi (Wright coefficient)
    numerator = np.sum((y1 - y2) ** 2)

    denominator = np.sum(y2**2)
    rchi = np.sqrt(numerator / denominator)

    return rchi, common_x, y1, y2


def homogeniety_checker(
    atoms,
    grid_density,
    slide_steps=2,
    target_species="all",
    upper_bound=1.5,
    lower_bound=0.5,
    box_threshold=0.1,
    seperated_species_threshold=0.5,
):
    atoms.wrap()
    species = np.unique(atoms.get_chemical_symbols()) if target_species == "all" else [target_species]

    if len(species) == 0:
        raise ValueError("No species found in the atoms object.")

    cell_lengths = np.array(atoms.get_cell()).diagonal()
    num_boxes = np.prod(grid_density)

    x_edges = np.linspace(0, cell_lengths[0], grid_density[0] + 1)
    y_edges = np.linspace(0, cell_lengths[1], grid_density[1] + 1)
    z_edges = np.linspace(0, cell_lengths[2], grid_density[2] + 1)

    slide_fractions = np.linspace(0, 1, slide_steps, endpoint=False)

    phase_seperated_species = 0

    for spec in species:
        spec_atoms = atoms[np.array(atoms.get_chemical_symbols()) == spec]
        num_atoms = len(spec_atoms)
        avg_atoms_per_box = num_atoms / num_boxes

        if avg_atoms_per_box < 2:
            continue

        positions = spec_atoms.get_positions()

        out_of_bounds_boxes = []
        for dx, dy, dz in product(slide_fractions, repeat=3):
            # Apply offset in each direction
            x_slide = x_edges + dx * (cell_lengths[0] / grid_density[0])
            y_slide = y_edges + dy * (cell_lengths[1] / grid_density[1])
            z_slide = z_edges + dz * (cell_lengths[2] / grid_density[2])

            x_slide[-1] = cell_lengths[0]
            y_slide[-1] = cell_lengths[1]
            z_slide[-1] = cell_lengths[2]

            x_idx = np.digitize(positions[:, 0], x_slide) - 1
            y_idx = np.digitize(positions[:, 1], y_slide) - 1
            z_idx = np.digitize(positions[:, 2], z_slide) - 1

            x_idx[x_idx == -1] = 2
            y_idx[y_idx == -1] = 2
            z_idx[z_idx == -1] = 2

            counts = np.zeros(grid_density, dtype=int)
            for xi, yi, zi in zip(x_idx, y_idx, z_idx):
                counts[xi, yi, zi] += 1

            too_low = counts < avg_atoms_per_box * lower_bound
            too_high = counts > avg_atoms_per_box * upper_bound
            out_of_bounds_boxes.append(np.sum(too_low | too_high))

        if np.sum(out_of_bounds_boxes) > num_boxes * slide_steps**3 * box_threshold:
            phase_seperated_species += 1

    if phase_seperated_species > seperated_species_threshold * len(species):
        return False
    else:
        return True


def dimer_checker(atoms, bond_length=2.0, num_allowed=2):
    atoms.wrap()
    species = ["O", "N", "F", "Cl", "Br", "I"]
    for spec in species:
        if spec not in atoms.get_chemical_symbols():
            continue
        spec_atoms = atoms[np.array(atoms.get_chemical_symbols()) == spec]
        if len(spec_atoms) < 2:
            continue
        spec_positions = spec_atoms.get_positions()
        distances = np.linalg.norm(spec_positions[:, np.newaxis] - spec_positions, axis=-1)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        num_dimers = np.sum(distances < bond_length)
        if num_dimers > num_allowed:  # Adjust threshold as needed
            return True
    return False
