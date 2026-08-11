import numpy as np

from vitrum.geometry import require_orthorhombic


def get_high_low_displacement_index(initial_state, current_state, target_atom, percentage=0.25):
    """
    Calculates the indices of the atoms with the high and low displacements between an initial and current state.

    Parameters:
        initial_state (Atoms): The initial state of the system.
        current_state (Atoms): The current state of the system.
        target_atom (str or int): The chemical symbol or atomic number of the target atom.
        percentage (float, optional): The percentage of the highest and lowest displacements to consider. Defaults to
            0.25.

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

    Steps between frames are taken in fractional coordinates and converted back with the
    current frame's cell, so a trajectory whose cell changes over time (NPT) unwraps
    correctly: an atom held at fixed fractional coordinates does not move.

    Parameters:
        atoms_list (list of Atoms objects): The list of Atoms objects to unwrap.

    Returns:
        unwrapped_atoms_list (list of Atoms objects): The unwrapped list of Atoms objects.

    Raises:
        ValueError: If atoms_list is empty.
        NotImplementedError: If any frame has a non-orthorhombic cell.
    """

    if not atoms_list or len(atoms_list) == 0:
        raise ValueError("The input atoms_list must be a non-empty list of ASE Atom objects.")
    unwrapped_atoms_list = [atoms.copy() for atoms in atoms_list]

    position = unwrapped_atoms_list[0].get_positions()
    previous_fractional = unwrapped_atoms_list[0].get_scaled_positions(wrap=False)
    for atoms in unwrapped_atoms_list:
        cell = require_orthorhombic(atoms.get_cell(), "unwrap_trajectory")
        fractional = atoms.get_scaled_positions(wrap=False)
        # The shortest fractional step is the real one; a longer one crossed a boundary.
        step = fractional - previous_fractional
        step -= np.round(step)
        position = position + step * cell
        previous_fractional = fractional
        atoms.set_positions(position)
    return unwrapped_atoms_list
