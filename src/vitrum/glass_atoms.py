"""Deprecated `Atoms` subclass, kept for one release.

`GlassAtoms` was originally the home for analysis that operates on a single structure. That
logic now lives on the analysis classes, which take and hold plain `ase.Atoms`, so there is
no longer a second atoms type to know about. This module remains only so that existing
scripts keep working; every method delegates to its replacement and warns.

Deprecated since 1.1.0, to be removed in 2.0.0. See `docs/vitrum/glass_atoms.md` for the
old-to-new mapping.
"""

import warnings
from numbers import Integral

import numpy as np
from ase import Atoms

from vitrum.coordination import Coordination
from vitrum.geometry import distance_matrix, partial_pdf
from vitrum.io_helpers import correct_atom_types, get_density


def _deprecated(old: str, new: str) -> None:
    """Emit the standard deprecation warning for a `GlassAtoms` method."""
    warnings.warn(
        f"{old} is deprecated and will be removed in vitrum 2.0.0. Use {new} instead; "
        "it takes a plain ase.Atoms object.",
        DeprecationWarning,
        stacklevel=3,
    )


class GlassAtoms(Atoms):
    """
    Deprecated extended ASE Atoms class for glass structure analysis.

    Deprecated since 1.1.0, to be removed in 2.0.0. Pass plain `ase.Atoms` objects to
    `Coordination`, `Scattering` and the other analysis classes instead.
    """

    def get_dist(self) -> np.ndarray:
        """
        Deprecated. Use `vitrum.geometry.distance_matrix(atoms)`.

        Returns:
            np.ndarray: An array of shape (n_atoms, n_atoms) containing the distances
                between each pair of atoms.

        Raises:
            NotImplementedError: If the cell is not orthorhombic.
        """
        _deprecated("GlassAtoms.get_dist", "vitrum.geometry.distance_matrix(atoms)")
        return distance_matrix(self, "GlassAtoms.get_dist")

    def set_new_chemical_symbols(self, symbol_map: dict[int, str]):
        """
        Deprecated. Use `vitrum.io_helpers.correct_atom_types(atoms, symbol_map)`.

        Args:
            symbol_map (dict[int, str]): A dictionary mapping atomic numbers to new chemical symbols.
        """
        _deprecated(
            "GlassAtoms.set_new_chemical_symbols",
            "vitrum.io_helpers.correct_atom_types(atoms, symbol_map)",
        )
        correct_atom_types(self, symbol_map)

    def get_pdf(self, target_atoms, rrange=10, nbin=100, indicies=None):
        """
        Deprecated. Use `Scattering(atoms).get_partial_pdf(pair)`, or
        `vitrum.geometry.partial_pdf` for one pair from a precomputed distance matrix.

        Args:
            target_atoms (list[str | int]): A list of two elements representing the target atoms.
                Each element can be either a string (chemical symbol) or an integer (atomic number).
            rrange (float, optional): The range within which to calculate the PDF. Defaults to 10.0.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 100.
            indicies (list[np.ndarray], optional): A list of two arrays representing the indices
                of the target atoms. Specifying this parameter will override the target_atoms parameter.
                Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - xval: An array of shape (nbin,) containing the distance values.
                - pdf: An array of shape (nbin,) containing the PDF values.
        """
        _deprecated("GlassAtoms.get_pdf", "Scattering(atoms).get_partial_pdf(pair)")

        if indicies is None:
            if isinstance(target_atoms[0], str):
                symbols = self.get_chemical_symbols()
            elif isinstance(target_atoms[0], Integral):
                symbols = self.get_atomic_numbers()
            else:
                raise TypeError("target_atoms must contain strings or integers.")
        else:
            symbols = None

        return partial_pdf(
            distance_matrix(self, "GlassAtoms.get_pdf"),
            symbols,
            self.get_volume(),
            target_atoms,
            rrange,
            nbin,
            indices=indicies,
        )

    def get_all_angles(
        self,
        center_type: str,
        neigh_types: str | list[str],
        cutoff: float | dict | str = "Auto"
    ) -> list[np.ndarray]:
        """
        Deprecated. Use `Coordination([atoms]).get_angles(...)`.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (str | list[str]): The atomic symbol(s) of the neighbor atoms,
                either one symbol or a list of exactly two.
            cutoff (float | dict | str, optional): Range within which to calculate the
                angular distribution. Defaults to "Auto". Takes the same spellings as
                `Coordination`; the two arms of a same-species angle share one cutoff.

        Returns:
            list[np.ndarray]: A list of arrays containing the angular distribution values.

        Raises:
            ValueError: If center_type or neigh_types are not present in the structure, or
                if neigh_types does not have exactly one or two entries.
        """
        _deprecated("GlassAtoms.get_all_angles", "Coordination([atoms]).get_angles(...)")
        return Coordination([self]).get_angles(
            center_type, neigh_types, cutoff, per_atom=True
        )[0]

    def get_coordination_number(
        self,
        center_type: str,
        neigh_type: str,
        cutoff: float | str = "Auto"
    ) -> list[int]:
        """
        Deprecated. Use `Coordination([atoms]).get_coordination_numbers(..., per_atom=True)`.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (str): The atomic symbol of the neighbor atoms.
            cutoff (float | str, optional): The range within which to calculate the
                coordination number. Defaults to "Auto".

        Returns:
            list[int]: A list containing the coordination numbers for each center atom.

        Raises:
            ValueError: If center_type or neigh_type is not present in the structure.
        """
        _deprecated(
            "GlassAtoms.get_coordination_number",
            "Coordination([atoms]).get_coordination_numbers(..., per_atom=True)",
        )
        return Coordination([self]).get_coordination_numbers(
            center_type, neigh_type, cutoff, per_atom=True
        )[0].tolist()

    def get_bridging_analysis(
        self,
        center_type: str,
        bridge_type: str,
        former_types: list[str] | None = None,
        cutoff: float | str = "Auto",
    ) -> list[int]:
        """
        Deprecated. Use `Coordination([atoms]).get_bridging_analysis(...)`.

        Args:
            center_type (str): The type of the center atoms.
            bridge_type (str): The type of the bridge atoms.
            former_types (list[str], optional): A list of types of the former atoms. Defaults to None.
            cutoff (float | str, optional): The cutoff distance for considering a bridge.
                If "Auto", the cutoff is determined by finding the minimum value after the peak in the
                radial distribution function. If a float or int, the cutoff is set to the specified value.
                Defaults to "Auto".

        Returns:
            list[int]: A list of the number of bridges for each center atom.

        Raises:
            ValueError: If center_type, bridge_type or any former_type is absent.
            TypeError: If former_types is neither None nor a list.
        """
        _deprecated(
            "GlassAtoms.get_bridging_analysis",
            "Coordination([atoms]).get_bridging_analysis(...)",
        )
        return Coordination([self]).get_bridging_analysis(
            center_type, bridge_type, former_types, cutoff, per_atom=True
        )[0].tolist()

    def get_density(self) -> float:
        """
        Deprecated. Use `vitrum.io_helpers.get_density(atoms)`.

        Returns:
            float: Density in g/cm^3.
        """
        _deprecated("GlassAtoms.get_density", "vitrum.io_helpers.get_density(atoms)")
        return get_density(self)

    def get_neighbors(
        self, center_type: str, cutoff: float | dict
    ) -> dict[str, list[np.ndarray]]:
        """
        Deprecated. Use `Coordination([atoms]).get_neighbors(...)`.

        Args:
            center_type (str): The type of the center atoms.
            cutoff (float | dict): The cutoff distance for considering a neighbor.
                If a float or int, the cutoff is set to the specified value.
                If a dictionary, it should map atom types to their respective cutoffs.

        Returns:
            dict[str, list[np.ndarray]]: A dict mapping each neighbor species to a list with
                one entry per center atom. Each entry is an array of positional indices into
                that species' atoms (i.e. indices into ``np.where(types == neigh_type)[0]``),
                not global atom indices. `Coordination.get_neighbors` returns global indices
                instead; this method keeps the original convention.

        Raises:
            ValueError: If center_type is not present in the structure.
            KeyError: If cutoff is a dict missing an entry for a species.
            TypeError: If cutoff is not a float, int, or dict.
        """
        _deprecated("GlassAtoms.get_neighbors", "Coordination([atoms]).get_neighbors(...)")
        neighbors = Coordination([self]).get_neighbors(center_type, cutoff)[0]

        # Convert the global indices back to the species-relative ones this method has
        # always returned. np.where yields sorted indices and every entry is a sorted
        # subset of them, so searchsorted inverts the mapping exactly.
        types = np.array(self.get_chemical_symbols())
        return {
            species: [
                np.searchsorted(np.where(types == species)[0], entry) for entry in per_center
            ]
            for species, per_center in neighbors.items()
        }
