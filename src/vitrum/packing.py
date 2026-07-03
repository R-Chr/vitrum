import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.symbols import symbols2numbers
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.core import Composition, Structure
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

from vitrum.volume_estimation import get_volume


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
                raise RuntimeError(
                    f"Overlap relaxation invariant violated: max residual {np.max(ds):.3e} > 1e-10"
                )
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
