"""Cell-volume estimation, and the per-atom radii the estimators are built on."""

import warnings

import numpy as np
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.symbols import symbols2numbers
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element, Species
from scipy.constants import Avogadro

IONIC_PACKING_FRACTION = 0.48
COVALENT_PACKING_FRACTION = 1 / 3

_EXTRA_HINT = "requires the optional volume_estimation extra: pip install vitrum[volume_estimation]"


def get_packing_radii(elements, composition, source="covalent"):
    """Per-atom radii in Å. source: 'covalent' | 'atomic' | 'ionic'."""
    if source == "covalent":
        return covalent_radii[symbols2numbers(elements)].copy()

    if source == "atomic":
        return np.array(
            [
                Element(el).atomic_radius or covalent_radii[symbols2numbers([el])[0]]  # fallback: covalent
                for el in elements
            ]
        )

    if source == "ionic":
        oxi = guess_oxi_states(composition)
        if not oxi:
            return covalent_radii[symbols2numbers(elements)].copy()
        radii = []
        for el in elements:
            state = round(oxi[el])
            try:
                r = Species(el, state).ionic_radius
            except (KeyError, ValueError):
                r = None
            radii.append(r or Element(el).atomic_radius or covalent_radii[symbols2numbers([el])[0]])
        return np.array(radii)

    raise ValueError(f"unknown radii source: {source}")


def guess_oxi_states(composition, max_exact_atoms=100, totals=(40, 60, 100)):
    """Guess oxidation states, rounding the composition first only if it's large.
    Returns {element: state} or None."""
    comp = composition.reduced_composition

    if comp.num_atoms <= max_exact_atoms:
        guesses = comp.oxi_state_guesses(max_sites=-1)
        if guesses:
            return {el: round(v) for el, v in guesses[0].items()}
        warnings.warn(f"No charge-balanced oxidation states for {comp.reduced_formula}; using covalent radii.")
        return None

    amts = comp.element_composition.get_el_amt_dict()
    n = sum(amts.values())
    for total in totals:
        approx = Composition({el: max(1, round(a / n * total)) for el, a in amts.items()})
        guesses = approx.oxi_state_guesses(max_sites=-1)
        if guesses:
            return {el: round(v) for el, v in guesses[0].items()}
    warnings.warn(
        f"Could not assign oxidation states for {comp.reduced_formula} "
        f"(rounded compositions not charge-balanceable); using covalent radii."
    )
    return None


def _import_atomate2_volume_helpers():
    """Import the atomate2 database helpers lazily.

    atomate2 is an optional extra, and `vitrum/__init__.py` reaches this module through
    `structure_gen` -> `packing`, so a module-scope import would make `import vitrum` fail
    wherever the extra is not installed.
    """
    try:
        from atomate2.common.jobs.mpmorph import (
            get_average_volume_from_db_cached,
            get_average_volume_from_mp,
        )
    except ImportError as e:
        raise ImportError(f"vol_per_atom_source='mp'/'icsd' {_EXTRA_HINT}") from e
    return get_average_volume_from_db_cached, get_average_volume_from_mp


def get_volume(
    composition: Composition | str,
    structure: dict,
    vol_per_atom_source: float | str = "ionic_radius",
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
            - "ionic_radius": Estimate from ionic radii and a calibrated packing fraction.
              Needs no network access or optional dependencies, which is why it is the default.
              Calibrated on oxides (see `IONIC_PACKING_FRACTION` in this module); for metallic or
              covalent systems no charge-balanced oxidation states are found, covalent radii are
              used instead and the calibration does not hold — prefer `density` or an explicit
              float there.
            - "mp": Use Materials Project (requires API key and the volume_estimation extra).
            - "icsd": Use ICSD database (requires the volume_estimation extra).
            - "density": Calculate from provided density.
            - "covalent_radius": Estimate from covalent radii. Poorly calibrated for ionic
              systems, where it is off by a factor of 0.4-3.2 across common oxides.
            - "convex_hull": Estimate from convex hull on Materials Project.
            - float: Directly provide the volume per atom.
            Defaults to "ionic_radius".
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

    if density is not None:
        if not isinstance(density, (float, int)) or isinstance(density, bool):
            raise ValueError("Density must be a float or int.")

        if vol_per_atom_source not in ("density", "ionic_radius"):
            raise ValueError(
                f"Got both density={density} and vol_per_atom_source={vol_per_atom_source!r}, "
                "which are two different ways of setting the cell volume. Pass only one "
                "(omit vol_per_atom_source to estimate the volume from the density)."
            )
        struct_db = "density"

    if struct_db == "density":
        if density is None:
            raise ValueError("vol_per_atom_source='density' requires a density (in g/cm^3).")
        # Total cell mass in g/mol -> grams -> cm^3 -> Angstrom^3.
        mass = float(np.sum([atomic_masses[atomic_numbers[el]] * count for el, count in structure.items()]))
        cell_vol = (mass / Avogadro / density) * 1e24

    elif isinstance(vol_per_atom_source, float | int):
        vol_per_atom = vol_per_atom_source

    elif struct_db == "mp":
        _, get_average_volume_from_mp = _import_atomate2_volume_helpers()
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)

    elif struct_db == "icsd":
        get_average_volume_from_db_cached, _ = _import_atomate2_volume_helpers()
        vol_per_atom = get_average_volume_from_db_cached(composition, db_name="icsd", **db_kwargs)

    elif struct_db == "ionic_radius":
        elements = sum([[key] * structure[key] for key in structure], [])
        all_radii = get_packing_radii(elements, Composition(composition), source="ionic")
        cell_vol = float(np.sum(4 / 3 * np.pi * all_radii**3)) / IONIC_PACKING_FRACTION

    elif struct_db == "covalent_radius":
        all_radii = np.hstack([np.repeat(covalent_radii[atomic_numbers[key]], structure[key]) for key in structure])
        cell_vol = float(np.sum(4 / 3 * np.pi * all_radii**3)) / COVALENT_PACKING_FRACTION

    elif struct_db == "convex_hull":
        try:
            from mp_api.client import MPRestError
        except ImportError as e:
            raise ImportError(f"vol_per_atom_source='convex_hull' {_EXTRA_HINT}") from e
        try:
            vol_per_atom = get_average_volume_convex_hull(composition, MP_API_KEY=MP_API_KEY)
        except MPRestError as e:
            raise ValueError(f"Could not retrieve volume from convex hull. Check your MP_API_KEY. Error: {e}")

    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source}.")

    if cell_vol is None:
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
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise ImportError(f"get_average_volume_convex_hull {_EXTRA_HINT}") from e

    with MPRester(api_key=MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements=[str(el) for el in composition.elements],
            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
        )
    pd = PhaseDiagram(entries)
    decomp = pd.get_decomposition(composition)
    volume = sum([d.structure.volume / d.composition.num_atoms * decomp[d] for d in decomp])
    return volume
