import numpy as np
from vitrum.glass_Atoms import GlassAtoms
import itertools
from typing import List, Union, Optional, Tuple, Any
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

    def get_coordination_numbers(self, center_type: str, neigh_type: str, nbin: int = 70, cutoff: str = "Auto"):
        """
        Not implemented yet.
        """
        pass

# Alias for backward compatibility
coordination = Coordination
