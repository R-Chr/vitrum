import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from atomate2.common.jobs.mpmorph import (
    get_average_volume_from_db_cached,
    get_average_volume_from_mp,
)
from mp_api.client import MPRester, MPRestError
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition


def get_volume(
    composition: Composition | str,
    structure: dict,
    vol_per_atom_source: float | str = "mp",
    db_kwargs: dict | None = None,
    density: float | None = None,
    MP_API_KEY: str | None = None,
):
    """
    Get the volume of the cell based on the composition and various estimation methods.

    Args:
        composition (Union[Composition, str]): The composition of the material.
        structure (dict): A dictionary mapping element symbols to their count in the structure.
        vol_per_atom_source (Union[float, str], optional): Method to estimate volume per atom.
            Options:
            - "mp": Use Materials Project (requires API key).
            - "icsd": Use ICSD database (if available).
            - "density": Calculate from provided density.
            - "covalent_radius": Estimate from covalent radii.
            - "convex_hull": Estimate from convex hull on Materials Project.
            - float: Directly provide the volume per atom.
            Defaults to "mp".
        db_kwargs (dict, optional): Keyword arguments for database queries. Defaults to None.
        density (float, optional): Density in g/cm^3. Required if vol_per_atom_source="density". Defaults to None.
        MP_API_KEY (str, optional): Materials Project API key. Defaults to None.

    Returns:
        float: The calculated total volume of the cell in Angstrom^3.

    Raises:
        ValueError: If estimating from density but estimates fail, or if an unknown source is provided.
    """

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
    """
    Get the average volume per atom from the convex hull on Materials Project.

    Args:
        composition (Composition): The composition to query.
        MP_API_KEY (str, optional): Materials Project API Key.

    Returns:
        float: Average volume per atom.
    """
    with MPRester(api_key=MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements=[str(el) for el in composition.elements],
            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
        )
    pd = PhaseDiagram(entries)
    decomp = pd.get_decomposition(composition)
    volume = sum([d.structure.volume / d.composition.num_atoms * decomp[d] for d in decomp])
    return volume
