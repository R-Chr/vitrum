from collections import deque


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
