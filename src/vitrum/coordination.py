import numpy as np
from vitrum.glass_Atoms import GlassAtoms
import itertools
from typing import List, Union, Optional, Tuple, Dict
from ase import Atoms


class Coordination:
    """
    Class for analyzing coordination in glass structures.
    """
    def __init__(self, atoms_list: List[Atoms]):
        """
        Initializes a new instance of the coordination class with a list of atoms.

        Args:
            atoms_list (List[Atoms]): A list of atoms to be used for the coordination analysis.
        """

        self.atoms_list = [GlassAtoms(atom) for atom in atoms_list]
        self.chemical_symbols = atoms_list[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)

    def get_angle_distribution(
        self,
        center_type: str,
        neigh_types: Union[str, List[str]],
        nbin: int = 70,
        cutoff: Union[float, int, List[float], str] = "Auto",
        range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (Union[str, List[str]]): The atomic symbols of the neighbor atoms.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 70.
            cutoff (Union[float, int, List[float], str], optional): Range within which to calculate the angular distribution.
              Defaults to "Auto". Can be a list of cutoffs for each neighbor type, or a specific cutoff for all.
            range (Optional[Tuple[float, float]], optional): The range of the histogram.
              Defaults to None (range is determined automatically from np.histogram).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - angles: An array of shape (nbin,) containing the angle values (centers of bins).
                - dist: An array containing the angular distribution values (probability density).
        """
        angles_all = []
        for atoms in self.atoms_list:
            angles = atoms.get_all_angles(center_type, neigh_types, cutoff)
            flat_list_angles = list(itertools.chain(*angles))
            angles_all.append(np.array(flat_list_angles))

        dist, edges = np.histogram(np.hstack(angles_all), bins=nbin, density=True, range=range)
        angles = edges[1:] - 0.5 * (np.ptp(edges) / nbin)
        return angles, dist

    def get_coordination_numbers(
    self,
    center_type: str,
    neigh_type: Union[str, List[str]],
    cutoff: Union[float, int, List[float], str] = "Auto"
) -> Dict[int, float]:
    """
    Calculate the coordination number distribution over multiple frames.

    Args:
        center_type (str): The atomic symbol of the central atom.
        neigh_type (Union[str, List[str]]): The atomic symbol(s) of the
            neighbor atoms. Can be a single string (e.g. "O") or a list
            (e.g. ["O", "F"]) to count all neighbor types together.
        cutoff (Union[float, int, List[float], str], optional): The cutoff
            distance. If a list, must match the length of neigh_type.
            If "Auto", determined once from the first frame's PDF for each
            neigh_type. Defaults to "Auto".

    Returns:
        Dict[int, float]: A dictionary mapping each coordination number
            to its fraction. Returns an empty dict if atoms_list is empty
            or if center_type / neigh_type is absent from the structure.

    Raises:
        ValueError: If center_type or any neigh_type is not found in the
            structure, or if cutoff list length does not match neigh_type.
    """
    if len(self.atoms_list) == 0:
        return {}

    # normalise neigh_type to always be a list
    if isinstance(neigh_type, str):
        neigh_types = [neigh_type]
    else:
        neigh_types = list(neigh_type)

    species = np.unique(self.atoms_list[0].get_chemical_symbols())

    if center_type not in species:
        raise ValueError(
            f"center_type '{center_type}' not found in structure. "
            f"Available species: {list(species)}"
        )
    for nt in neigh_types:
        if nt not in species:
            raise ValueError(
                f"neigh_type '{nt}' not found in structure. "
                f"Available species: {list(species)}"
            )

    # normalise cutoff to a list matching neigh_types
    if cutoff == "Auto":
        from vitrum.utility import find_min_after_peak
        cutoffs = []
        for nt in neigh_types:
            pdf_r, pdf_g = self.atoms_list[0].get_pdf(
                target_atoms=[center_type, nt]
            )
            cutoffs.append(float(pdf_r[find_min_after_peak(pdf_g)]))
    elif isinstance(cutoff, (float, int)):
        cutoffs = [float(cutoff)] * len(neigh_types)
    elif isinstance(cutoff, list):
        if len(cutoff) != len(neigh_types):
            raise ValueError(
                f"cutoff list length ({len(cutoff)}) must match "
                f"neigh_type list length ({len(neigh_types)})."
            )
        cutoffs = [float(c) for c in cutoff]

    cn_all = []
    for atoms in self.atoms_list:
        # sum coordination numbers across all neigh_types per center atom
        cn_combined = None
        for nt, co in zip(neigh_types, cutoffs):
            cn = np.array(atoms.get_coordination_number(center_type, nt, co))
            cn_combined = cn if cn_combined is None else cn_combined + cn
        if cn_combined is not None:
            cn_all.extend(cn_combined.tolist())

    cn_all = np.array(cn_all)

    if len(cn_all) == 0:
        return {}

    cn_values, counts = np.unique(cn_all, return_counts=True)
    fractions = counts / counts.sum()
    return dict(zip(cn_values.tolist(), fractions.tolist()))
    
# Alias for backward compatibility
coordination = Coordination
