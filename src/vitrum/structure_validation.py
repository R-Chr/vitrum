from itertools import product

import numpy as np


def homogeneity_checker(
    atoms,
    grid_density,
    slide_steps=2,
    target_species="all",
    upper_bound=1.5,
    lower_bound=0.5,
    box_threshold=0.1,
    separated_species_threshold=0.5,
):
    """
    Check the homogeneity of the atomic structure by analyzing atom density in grid boxes.

    Args:
        atoms (Atoms): The ASE Atoms object to analyze.
        grid_density (tuple or list): Number of grid divisions in x, y, z directions (nx, ny, nz).
        slide_steps (int, optional): Number of steps to slide the grid for averaging. Defaults to 2.
        target_species (str or list, optional): Species to check. "all" checks all present species. Defaults to "all".
        upper_bound (float, optional): Multiplier for average density to consider a box over-dense. Defaults to 1.5.
        lower_bound (float, optional): Multiplier for average density to consider a box under-dense. Defaults to 0.5.
        box_threshold (float, optional): Fraction of boxes allowed to be out of bounds before flagging a species. Defaults to 0.1.
        separated_species_threshold (float, optional): Fraction of species allowed to be phase separated before the structure is flagged. Defaults to 0.5.

    Returns:
        bool: True if the structure is considered homogeneous, False otherwise.

    Raises:
        ValueError: If no species are found in the atoms object.
    """
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

    if phase_seperated_species > separated_species_threshold * len(species):
        return False
    else:
        return True


def dimer_checker(atoms, bond_length=2.0, num_allowed=2):
    """
    Check for the presence of dimers (e.g., O2, N2, F2, Cl2, Br2, I2) in the structure.

    Args:
        atoms (Atoms): The ASE Atoms object to check.
        bond_length (float, optional): The cutoff distance to define a bond. Defaults to 2.0.
        num_allowed (int, optional): The maximum number of allowed dimers before returning True. Defaults to 2.

    Returns:
        bool: True if the number of dimers exceeds `num_allowed`, False otherwise.
    """
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
