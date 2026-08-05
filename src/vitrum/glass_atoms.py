from itertools import product
from numbers import Integral
from typing import Dict, List, Optional, Union

import numpy as np
from ase import Atoms

from vitrum.geometry import find_min_after_peak, pdf, radial_bins, require_orthorhombic
from vitrum.geometry import get_dist_numba as dist


class GlassAtoms(Atoms):
    """
    Extended ASE Atoms class for glass structure analysis.
    """
    # def __init__(self):
    #    super().__init__()

    def get_dist(self) -> np.ndarray:
        """
        Calculate the distances between all pairs of atoms in the Atoms object.

        Returns:
            np.ndarray: An array of shape (n_atoms, n_atoms) containing the distances
                between each pair of atoms.

        Raises:
            NotImplementedError: If the cell is not orthorhombic.
        """
        dim = require_orthorhombic(self.get_cell(), "GlassAtoms.get_dist")
        positions = self.get_positions()
        return dist(positions, dim)

    def _require_species(self, *symbols: str) -> np.ndarray:
        """
        Check that every given chemical symbol is present, and return the symbol array.

        Args:
            *symbols (str): Chemical symbols that must be present in the structure.

        Returns:
            np.ndarray: The structure's chemical symbols, as an array.

        Raises:
            ValueError: If any requested symbol is absent.
        """
        types = np.array(self.get_chemical_symbols())
        present = set(np.unique(types).tolist())
        missing = [s for s in symbols if s not in present]
        if missing:
            raise ValueError(
                f"Species {missing} not present in the structure. "
                f"Available species: {sorted(present)}."
            )
        return types

    def set_new_chemical_symbols(self, symbol_map: Dict[int, str]):
        """
        Set new chemical symbols for the atoms in the object.

        Args:
            symbol_map (Dict[int, str]): A dictionary mapping atomic numbers to new chemical symbols.
        """
        corr_symbols = [symbol_map[i] for i in self.get_atomic_numbers()]
        self.set_chemical_symbols(corr_symbols)

    def get_pdf(self, target_atoms, rrange=10, nbin=100, indicies=None):
        """
        Calculate the probability density function (PDF) of a given pair of target atoms within a specified range.

        Args:
            target_atoms (List[Union[str, int]]): A list of two elements representing the target atoms.
                Each element can be either a string (chemical symbol) or an integer (atomic number).
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 10.0.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.
            indicies (Optional[List[np.ndarray]], optional): A list of two arrays representing the indices
                of the target atoms. Specifying this parameter will override the target_atoms parameter.
                Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - xval: An array of shape (nbin,) containing the distance values.
                - pdf: An array of shape (nbin,) containing the PDF values.
        """
        if indicies is None:
            if isinstance(target_atoms[0], str):
                types = self.get_chemical_symbols()
            elif isinstance(target_atoms[0], Integral):
                types = self.get_atomic_numbers()
            else:
                raise TypeError("target_atoms must contain strings or integers.")

            types = np.array(types)
            distances = self.get_dist()
            atom_1 = np.where(types == target_atoms[0])[0]
            atom_2 = np.where(types == target_atoms[1])[0]
        else:
            atom_1 = indicies[0]
            atom_2 = indicies[1]
            distances = self.get_dist() # Needed if indicies are provided but distances not calculated locally

        if len(atom_1) == 0 or len(atom_2) == 0:
            # Handle case where one species is missing to avoid errors in np.ix_
            xval, _ = radial_bins(rrange, nbin)
            return xval, np.zeros(nbin)

        # A like pair draws from the same index set, so the submatrix contains the zero-distance
        # diagonal and each atom has only n - 1 distinct partners. A cross pair has neither.
        like_pair = np.array_equal(atom_1, atom_2)
        if like_pair:
            n_pairs = len(atom_1) * (len(atom_1) - 1)
        else:
            n_pairs = len(atom_1) * len(atom_2)

        dist_list = distances[np.ix_(atom_1, atom_2)]
        return pdf(
            dist_list,
            self.get_volume(),
            rrange,
            nbin,
            n_pairs=n_pairs,
            exclude_self=like_pair,
        )

    def get_all_angles(
        self,
        center_type: str,
        neigh_types: Union[str, List[str]],
        cutoff: Union[float, int, List[float], str] = "Auto"
    ) -> List[np.ndarray]:
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        The angle is measured between two neighbours of the central atom, so exactly two
        neighbour species define it: `neigh_types=["O", "Na"]` gives O-center-Na angles. A
        single string is taken to mean both arms of the angle.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (Union[str, List[str]]): The atomic symbol(s) of the neighbor atoms,
                either one symbol or a list of exactly two.
            cutoff (Union[float, int, List[float], str], optional): Range within which to calculate the angular distribution.
                Defaults to "Auto". Can be a list of one cutoff per neighbor type, or a single cutoff for both.

        Returns:
            List[np.ndarray]: A list of arrays containing the angular distribution values.

        Raises:
            ValueError: If center_type or neigh_types are not present in the structure, or
                if neigh_types or an explicit cutoff list does not have exactly two entries.
        """
        if isinstance(neigh_types, str):
            neigh_types = [neigh_types, neigh_types]
        elif len(neigh_types) != 2:
            raise ValueError(
                f"neigh_types must be a single symbol or exactly two, got {list(neigh_types)}. "
                "An angle is defined by the two neighbours it spans; call this once per pair "
                "of neighbour species."
            )

        types = self._require_species(center_type, *neigh_types)

        distances = self.get_dist()

        center_index = np.where(types == center_type)[0]
        neigh_index = [np.where(types == neigh_type)[0] for neigh_type in neigh_types]

        if cutoff == "Auto":
            pdf_vals = [self.get_pdf(target_atoms=[center_type, neigh_type]) for neigh_type in neigh_types]
            cutoff = [
                pdf_vals[i][0][find_min_after_peak(pdf_vals[i][1], f"{center_type}-{neigh_types[i]}")]
                for i in range(len(neigh_types))
            ]
        elif isinstance(cutoff, (float, int)):
            cutoff = [cutoff, cutoff]
        elif len(cutoff) != 2:
            raise ValueError(
                f"cutoff must be a single value or one per neighbour type (two), got {list(cutoff)}."
            )

        angles = []

        for center in center_index:

            neighbor1 = np.where(
                (distances[neigh_index[0], center] < cutoff[0]) & (distances[neigh_index[0], center] > 0)
            )[0]

            neighbor2 = np.where(
                (distances[neigh_index[1], center] < cutoff[1]) & (distances[neigh_index[1], center] > 0)
            )[0]

            unique_pairs = set()

            # Generate combinations
            for a, b in product(neigh_index[0][neighbor1], neigh_index[1][neighbor2]):
                if a != b:
                    unique_pairs.add(tuple(sorted((a, b))))
            combinations = np.array(list(unique_pairs))
            if combinations.shape[0] < 1:
                continue

            indicies = np.vstack((combinations[:, 0], np.full(len(combinations), center), combinations[:, 1])).T
            angles.append(self.get_angles(indicies, mic=True))
        return angles

    def get_coordination_number(
        self,
        center_type: str,
        neigh_type: str,
        cutoff: Union[float, int, str] = "Auto"
    ) -> List[int]:
        """
        Calculate the coordination number of a given pair of target atoms within a specified range.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (str): The atomic symbol of the neighbor atoms.
            cutoff (Union[float, int, str], optional): The range within which to calculate the coordination number.
              Defaults to "Auto".

        Returns:
            List[int]: A list containing the coordination numbers for each center atom.

        Raises:
            ValueError: If center_type or neigh_type is not present in the structure.
        """
        types = self._require_species(center_type, neigh_type)
        distances = self.get_dist()
        atom_1 = np.where(types == center_type)[0]
        atom_2 = np.where(types == neigh_type)[0]
        dist_list = distances[np.ix_(atom_1, atom_2)]

        if cutoff == "Auto":
            pdf_vals = self.get_pdf(target_atoms=[center_type, neigh_type])
            cutoff = pdf_vals[0][find_min_after_peak(pdf_vals[1], f"{center_type}-{neigh_type}")]

        coordination_numbers = []
        for center in range(len(atom_1)):
            neighbors = np.where((dist_list[center, :] < cutoff) & (dist_list[center, :] > 0))[0]
            coordination_numbers.append(neighbors.shape[0])
        return coordination_numbers

    def get_bridging_analysis(
        self,
        center_type: str,
        bridge_type: str,
        former_types: Optional[List[str]] = None,
        cutoff: Union[float, int, str] = "Auto",
    ) -> List[int]:
        """
        Calculate the number of bridges for each center atom of a given type.

        Args:
            center_type (str): The type of the center atoms.
            bridge_type (str): The type of the bridge atoms.
            former_types (Optional[List[str]], optional): A list of types of the former atoms. Defaults to None.
            cutoff (Union[float, int, str], optional): The cutoff distance for considering a bridge.
                If "Auto", the cutoff is determined by finding the minimum value after the peak in the
                radial distribution function. If a float or int, the cutoff is set to the specified value.
                Defaults to "Auto".

        Returns:
            List[int]: A list of the number of bridges for each center atom.

        Raises:
            ValueError: If center_type, bridge_type or any former_type is absent.
            TypeError: If former_types is neither None nor a list.
        """
        types = self._require_species(center_type, bridge_type, *(former_types or []))
        distances = self.get_dist()
        centers = np.where(types == center_type)[0]
        bridges = np.where(types == bridge_type)[0]
        dist_list = distances[np.ix_(centers, bridges)]

        if former_types is None:
            formers = centers
        elif isinstance(former_types, list):
            formers = np.hstack([np.where(types == t)[0] for t in former_types])
        else:
            raise TypeError("former_types must be either None or a List of atom types")

        if cutoff == "Auto":
            pdf_vals = self.get_pdf(target_atoms=[center_type, bridge_type])
            cutoff = pdf_vals[0][find_min_after_peak(pdf_vals[1], f"{center_type}-{bridge_type}")]

        num_of_bridges = []
        for cen_ind, _ in enumerate(centers):
            neighbors = np.where((dist_list[cen_ind, :] < cutoff) & (dist_list[cen_ind, :] > 0))[0]
            neighbors = [bridges[neighbor] for neighbor in neighbors]
            q_species = 0
            for neigh in neighbors:
                num_bridges = np.where((distances[formers, neigh] < cutoff) & (distances[formers, neigh] > 0))[0]
                if num_bridges.shape[0] >= 2:
                    q_species += 1
            num_of_bridges.append(q_species)

        return num_of_bridges

    def get_density(self) -> float:
        """
        Calculate the density of the structure.
        
        Returns:
            float: Density in g/cm^3.
        """
        return (np.sum(self.get_masses()) / 6.02214076 * 10**-23) / (self.get_volume() * 10**-24)


    def get_neighbors(
        self, center_type: str, cutoff: Union[float, int, dict]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Find the neighbors of each center atom of a given type, grouped by neighbor species.

        Args:
            center_type (str): The type of the center atoms.
            cutoff (Union[float, int, dict]): The cutoff distance for considering a neighbor.
                If a float or int, the cutoff is set to the specified value.
                If a dictionary, it should map atom types to their respective cutoffs.

        Returns:
            Dict[str, List[np.ndarray]]: A dict mapping each neighbor species to a list with
                one entry per center atom. Each entry is an array of positional indices into
                that species' atoms (i.e. indices into ``np.where(types == neigh_type)[0]``),
                not global atom indices.

        Raises:
            ValueError: If center_type is not present in the structure.
            KeyError: If cutoff is a dict missing an entry for a species.
            TypeError: If cutoff is not a float, int, or dict.
        """

        types = self._require_species(center_type)
        atom_types = np.unique(types)
        distances = self.get_dist()

        index = {t: np.where(types == t)[0] for t in atom_types}

        def get_cutoff(neigh_type, cutoff):
            if isinstance(cutoff, dict):
                if neigh_type not in cutoff:
                    raise KeyError(f"No cutoff defined for atom type '{neigh_type}'")
                return cutoff[neigh_type]
            elif isinstance(cutoff, (float, int)):
                return cutoff
            else:
                raise TypeError("Cutoff must be either a float, int, or a dictionary mapping atom types to cutoffs.")
   
        neighbors = {}
        for neigh_type in atom_types:
            c = get_cutoff(neigh_type, cutoff)
            d = distances[np.ix_(index[neigh_type], index[center_type])]
            mask = (d < c) & (d > 0)
            neighbors[neigh_type] = [np.where(mask[:, i])[0] for i in range(len(index[center_type]))]

        return neighbors
