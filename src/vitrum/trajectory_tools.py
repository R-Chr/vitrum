import numpy as np


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
    index = np.where(np.array(initial_state.get_chemical_symbols()) == target_atom)[0]
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
