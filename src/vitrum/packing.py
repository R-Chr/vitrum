import warnings
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.core import Composition, Structure
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)
from scipy.stats import qmc

from vitrum.volume_estimation import get_packing_radii, get_volume


def get_random_packed(
    composition: str | dict | Composition,
    target_atoms: int = 100,
    min_distance: float | None = None,
    radii_source: str = "ionic",
    radii_scaling: float = 1.0,
    volume_scaling: float = 1.0,
    vol_per_atom_source: float | str = "ionic_radius",
    datatype: str = "ase",
    db_kwargs: dict | None = None,
    density: float | None = None,
    seed: int | None = None,
    side_ratios: list | None = None,
    algorithm: str = "sobol",
    **kwargs: Any,
) -> Atoms | Structure:
    """
    Generate a random packed structure based on the given composition.

    Parameters:
        composition (str, dict or pymatgen.core.Composition): The composition of the structure.
        density (float, optional): The target density of the structure (in g/cm^3). If not provided, the volume per atom
                                   is estimated using `vol_per_atom_source`.
        target_atoms (int, optional): The target number of atoms in the structure. Defaults to 100.
        min_distance (float, optional): Minimum distance between atoms in the structure. If not provided, no minimum
            distance is enforced.
        radii_source (str, optional): The source for the atomic radii. Can be "covalent", "atomic", or "ionic". Defaults
            to "ionic".
        radii_scaling (float, optional): Scaling factor for the covalent radii of the atoms. Defaults to 1.0.
        volume_scaling (float, optional): Scaling factor for the volume of the structure. Defaults to 1.0.
        vol_per_atom_source (float or str, optional): The source for the volume per atom. Either a float, or one of:
            "ionic_radius" (estimate from ionic radii; needs no network access or optional dependencies, calibrated
            on oxides), "mp" (Materials Project), "icsd" (Inorganic Crystal Structure Database), "density" (use the
            provided density), "covalent_radius" (estimate from covalent radii; poorly calibrated for ionic systems),
            or "convex_hull" (estimate from convex hull). Defaults to "ionic_radius".
            See `vitrum.volume_estimation.get_volume` for details.
        datatype (str, optional): The type of data to return. Can be "ase" for ASE format or "pymatgen"
                                  for pymatgen format. Defaults to "ase".
        db_kwargs (dict, optional): Additional keyword arguments for database access. Defaults to None.
        seed (int, optional): The seed for random number generation. Defaults to None.
        side_ratios (list, optional): The side ratios for the lattice. Defaults to None, i.e. a cubic cell.
        algorithm (str, optional): How to place the initial points, "sobol" or "random".
                                   Defaults to "sobol".

    Returns:
        data (ase.Atoms or pymatgen.core.Structure): The generated random packed structure.
    """

    side_ratios = [1, 1, 1] if side_ratios is None else side_ratios
    composition = Composition(composition)
    formula, factor = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(target_atoms / integer_composition.num_atoms)

    structure: dict[str, int] = {}
    for el in full_cell_composition:
        structure[str(el)] = int(full_cell_composition.element_composition.get(el))
    symbols = sum([[i] * structure[i] for i in structure], [])
    rng = np.random.default_rng(seed)

    cell_vol = get_volume(composition, structure, vol_per_atom_source, db_kwargs, density, **kwargs)

    cell_vol *= volume_scaling
    k = (cell_vol / (side_ratios[0] * side_ratios[1] * side_ratios[2])) ** (1 / 3)
    cell = np.array([side_ratios[0] * k, side_ratios[1] * k, side_ratios[2] * k])
    cell = np.diag(cell)

    radii = get_packing_radii(symbols, composition, source=radii_source) * radii_scaling

    if min_distance:
        radii = np.maximum(radii, min_distance / 2)
    nat = len(symbols)

    if algorithm == "sobol":
        frac = qmc.Sobol(d=3, seed=seed).random(nat)
        pos = frac @ cell
    elif algorithm == "random":
        pos = rng.random((nat, 3)) @ cell
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose 'sobol' or 'random'.")

    ats = Atoms(symbols, cell=cell, pbc=True, positions=pos)
    skin_init = 0.2  # Å of extra buffer at the start (~10% of a typical radius)
    decay_iters = 50  # skin reaches zero by this iteration

    nl = NeighborList(radii + skin_init / 2, self_interaction=False, bothways=True, skin=0.3)

    for it in range(500):
        skin = skin_init * max(0.0, 1.0 - it / decay_iters)  # linear decay
        nl.update(ats)
        pos = ats.get_positions()
        dx = np.zeros_like(pos)
        dsum = 0.0
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            if len(indices) == 0:
                continue
            rs = pos[indices] + offsets @ cell - pos[i]
            d = np.linalg.norm(rs, axis=1)
            coincident = d < 1e-12
            if np.any(coincident):
                rs[coincident] = rng.normal(size=(int(coincident.sum()), 3))
                d[coincident] = np.linalg.norm(rs[coincident], axis=1)
            ds_true = d - (radii[indices] + radii[i])
            dsum += ds_true[ds_true < 0].sum()
            ds_eff = np.minimum(ds_true - skin, 0.0)
            dx[i] = np.sum(rs / d[:, None] * ds_eff[:, None], axis=0)
        ats.set_positions(pos + dx)
        if dsum >= -1.0e-5:
            break
    else:
        warnings.warn(f"Cell packing not converged after 500 iterations, final overlap sum {dsum:.3e}")

    ats.wrap()
    if datatype != "pymatgen":
        return ats
    return Structure(
        lattice=ats.get_cell(),
        species=ats.get_chemical_symbols(),
        coords=ats.get_scaled_positions(),
        to_unit_cell=True,
        coords_are_cartesian=False,
    )


def apply_strain_to_structure(structure: Structure, deformations: list) -> list:
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
