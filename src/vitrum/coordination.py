import itertools
from collections.abc import Sequence
from numbers import Real

import numpy as np
from ase import Atoms

from vitrum.bonds import Bonds, _as_list, _Frame
from vitrum.geometry import find_min_after_peak

#: How every method in this module accepts a cutoff. A cutoff belongs to a bond, so a dict
#: is keyed by one: ``{("Si", "O"): 1.9}``, or ``{"O": 1.9}`` as shorthand for the bond from
#: the centre to that species.
Cutoff = float | str | dict[str | tuple[str, str], float]


_PDF_RANGE = 6.0
_PDF_BIN_WIDTH = 0.1
_EXTREMA_ORDER = 4


def _auto_cutoff(frame: _Frame, pair: tuple[str, str]) -> float:
    """
    A cutoff taken from the first minimum after the first peak of the pair's PDF.

    Raises:
        ValueError: If the PDF has no local minimum after a first peak, or if that
            minimum sits too near the end of `_PDF_RANGE` to be trusted.
    """
    context = f"{pair[0]}-{pair[1]}"
    bins = round(_PDF_RANGE / _PDF_BIN_WIDTH)
    pdf_r, pdf_g = frame.partial_pdf(pair, _PDF_RANGE, bins)

    index = find_min_after_peak(pdf_g, context)
    if index >= bins - _EXTREMA_ORDER:
        raise ValueError(
            f"{context}: first PDF minimum falls at the edge of the {_PDF_RANGE} Å "
            f"range and may be spurious; set this cutoff explicitly."
        )
    return float(pdf_r[index])


def _cutoff_for(cutoff: dict, pair: tuple[str, str]) -> float:
    """The dict entry for one bond: the pair either way round, or the neighbour alone."""
    center, neigh = pair
    for key in ((center, neigh), (neigh, center), neigh):
        if key in cutoff:
            return float(cutoff[key])
    raise KeyError(
        f"No cutoff defined for atom type '{neigh}'. The {center}-{neigh} bond needs an "
        f"entry keyed {neigh!r} or {pair!r}."
    )


def _auto_cutoffs(frame: _Frame, pairs: Sequence[tuple[str, str]]) -> list[float]:
    """One "Auto" cutoff per bond, taking each distinct bond's PDF only once."""
    resolved = {}
    for pair in pairs:
        if frozenset(pair) not in resolved:
            resolved[frozenset(pair)] = _auto_cutoff(frame, pair)
    return [resolved[frozenset(pair)] for pair in pairs]


def _resolve_cutoffs(
    frame: _Frame,
    pairs: Sequence[tuple[str, str]],
    cutoff: Cutoff,
) -> list[float]:
    """
    Turn any accepted `cutoff` into one float per bond, given the bonds as species pairs.

    The only place that understands "Auto", so the functions doing the counting never see
    anything but floats.

    Raises:
        TypeError: If `cutoff` is of an unusable type.
        ValueError: If a species is absent, or the string is not "Auto".
        KeyError: If a cutoff dict has no entry for one of the bonds.
    """
    # Validate up front: an absent species otherwise surfaces as an all-zero PDF with no
    # first minimum, reporting a failed cutoff search rather than the real problem.
    for pair in pairs:
        frame.require(*pair)

    if isinstance(cutoff, str):
        if cutoff == "Auto":
            return _auto_cutoffs(frame, pairs)
        raise ValueError(
            f"Invalid cutoff {cutoff!r}. The only string spelling is 'Auto', spelt exactly; "
            f"anything else has to be a number or a dict."
        )

    if isinstance(cutoff, Real) and not isinstance(cutoff, bool):
        return [float(cutoff)] * len(pairs)

    if isinstance(cutoff, dict):
        return [_cutoff_for(cutoff, pair) for pair in pairs]

    raise TypeError(
        f"Invalid cutoff {cutoff!r}. Expected the string 'Auto', a number, or a dict "
        f"keyed by bond as ('Si', 'O') or by neighbour species as 'O'."
    )


def _angle_arms(neigh_types: str | Sequence[str]) -> list[str]:
    """Normalise `neigh_types` to the two arms of an angle."""
    arms = _as_list(neigh_types)
    if len(arms) == 1:
        return [arms[0], arms[0]]
    if len(arms) != 2:
        raise ValueError(f"neigh_types must be a single symbol or exactly two, got {arms}.")
    return arms


def _angles(
    frame: _Frame,
    center_type: str,
    neigh_types: Sequence[str],
    cutoffs: Sequence[float],
) -> list[np.ndarray]:
    """
    Angles subtended at each atom of `center_type` by two of its neighbours.

    `neigh_types` and `cutoffs` both hold exactly two entries. Returns one array of angles
    per atom of `center_type`, in index order, empty where the atom has no qualifying
    neighbour pair, so the result stays aligned with `frame.index(center_type)`.
    """

    same_species = neigh_types[0] == neigh_types[1]
    arm_a = frame.bonds(center_type, neigh_types[0], cutoffs[0]).lists()
    arm_b = arm_a if same_species else frame.bonds(center_type, neigh_types[1], cutoffs[1]).lists()

    triples, sizes = [], []
    for center, a, b in zip(frame.index(center_type), arm_a, arm_b):
        if same_species:
            pairs = np.asarray(list(itertools.combinations(a, 2)))
        else:
            # Distinct species share no global index, so no pair can be an atom with itself.
            pairs = np.asarray(list(itertools.product(a, b)))
        sizes.append(len(pairs))
        if len(pairs):
            triples.append(np.column_stack((pairs[:, 0], np.full(len(pairs), center), pairs[:, 1])))

    if not triples:
        return [np.zeros(0) for _ in sizes]
    measured = frame.atoms.get_angles(np.vstack(triples), mic=True)
    return list(np.split(measured, np.cumsum(sizes)[:-1]))


def _bridging_speciation(frame: _Frame, bridge_type: str, former_cutoffs: dict[str, float]) -> np.ndarray:
    """Number of network formers bonded to each `bridge_type` atom."""
    return sum(frame.bonds(former, bridge_type, cut).degrees() for former, cut in former_cutoffs.items())


def _bridging_analysis(
    frame: _Frame,
    center_type: str,
    bridge_type: str,
    center_cutoff: float,
    former_cutoffs: dict[str, float],
) -> np.ndarray:
    """Number of bridging `bridge_type` atoms around each `center_type` atom."""
    formers_per_bridge = _bridging_speciation(frame, bridge_type, former_cutoffs)
    return frame.bonds(center_type, bridge_type, center_cutoff).select_neighs(formers_per_bridge >= 2).counts()


def _neighbors(frame: _Frame, center_type: str, cutoffs: dict[str, float]) -> dict[str, list[np.ndarray]]:
    """Global indices of each `center_type` atom's neighbours, grouped by species."""
    missing = [t for t in frame.species if t not in cutoffs]
    if missing:
        raise KeyError(
            f"No cutoff defined for atom type '{missing[0]}'. Cutoffs are resolved from the "
            f"first frame, so this frame holds a species the first one does not."
        )
    return {str(t): frame.bonds(center_type, t, cutoffs[t]).lists() for t in frame.species}


def _fractions(values: np.ndarray) -> dict[int, float]:
    """Map each distinct integer count in `values` to the fraction of atoms showing it."""
    values = np.asarray(values)
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return dict(zip(unique.tolist(), (counts / counts.sum()).tolist()))


class Coordination:
    """
    Class for analyzing coordination in glass structures.

    Frames are plain ASE `Atoms` objects and are stored as given.

    A cutoff belongs to a bond rather than to an atom, so every method accepts the same
    spellings: `"Auto"`, a number for every bond, `{("Si", "O"): 1.6}` per bond, or
    `{"O": 1.6}` as shorthand for the bond from the centre to that species. Unneeded keys
    are ignored, so one dict serves every method. It follows that the two arms of a
    same-species angle always share one cutoff.

    An "Auto" cutoff is resolved **once, from one frame**, and then applied to every frame,
    so it cannot drift along a trajectory and every method agrees on the same value. That
    frame is the first by default; frame 0 of an MD run is often the starting crystal or
    packed box, whose PDF minima can sit well away from the equilibrated liquid's, so
    `cutoff_frame` picks a more representative one. The first frame's composition is taken
    as representative regardless: it is what `chemical_symbols` and `species` describe.

    Attributes:
        atoms_list (List[Atoms]): The frames, as given.
        chemical_symbols (List[str]): The first frame's chemical symbols.
        species (np.ndarray): The distinct species present in the first frame.
        cutoff_frame (int): Index of the frame an "Auto" cutoff is resolved from.
    """

    def __init__(self, atoms_list: Atoms | list[Atoms], cutoff_frame: int = 0):
        """
        Initialize the analysis from one structure or a list of frames.

        Args:
            atoms_list (Union[Atoms, List[Atoms]]): A single structure, or a list of
                frames to aggregate over. A single `Atoms` is wrapped in a list rather
                than iterated, which would otherwise yield individual `Atom` objects.
            cutoff_frame (int, optional): Index of the frame an "Auto" cutoff is measured
                from, negative counting from the end. Defaults to 0, the first frame.

        Raises:
            ValueError: If no frames are given.
            TypeError: If any frame is not an `Atoms` object.
            IndexError: If `cutoff_frame` is out of range.
        """
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        frames = list(atoms_list)
        if not frames:
            raise ValueError("atoms_list must contain at least one Atoms object.")
        wrong = sorted({type(a).__name__ for a in frames if not isinstance(a, Atoms)})
        if wrong:
            raise TypeError(f"atoms_list must contain Atoms objects, got {wrong}.")
        if not -len(frames) <= cutoff_frame < len(frames):
            raise IndexError(f"cutoff_frame={cutoff_frame} is out of range for {len(frames)} frame(s).")
        self.atoms_list = frames
        self.chemical_symbols = frames[0].get_chemical_symbols()
        self.species = np.unique(self.chemical_symbols)
        self.cutoff_frame = cutoff_frame

    def _cutoffs(self, pairs: Sequence[tuple[str, str]], cutoff: Cutoff) -> list[float]:
        """Resolve `cutoff` to one float per bond in `pairs`, using `cutoff_frame`."""
        return _resolve_cutoffs(_Frame(self.atoms_list[self.cutoff_frame]), pairs, cutoff)

    def _frames(self, *required: str):
        """Yield one checked `_Frame` at a time, so peak memory stays at one frame's worth."""
        for atoms in self.atoms_list:
            frame = _Frame(atoms)
            frame.require(*required)
            yield frame

    def get_bonds(
        self, center_type: str | list[str], neigh_type: str | list[str], cutoff: Cutoff = "Auto"
    ) -> list[Bonds]:
        """
        Bonds between two species selections, for every frame.

        This is the primitive the rest of the class reduces: `get_coordination_numbers` is
        `counts()` on it, `get_neighbors` is `lists()`, and `get_bridging_analysis` is a
        `degrees()` test feeding a `select_neighs`.

        Args:
            center_type (Union[str, List[str]]): The species making up the centre selection.
            neigh_type (Union[str, List[str]]): The species making up the neighbour
                selection. May name the same species as `center_type`.
            cutoff (Cutoff, optional): Maximum separation for a bond, exclusive. One cutoff
                applies to the whole selection, so a multi-species selection has to resolve
                to a single value: give it a number, or a dict whose entries agree.
                Defaults to "Auto".

        Returns:
            List[Bonds]: One `Bonds` object per frame. Its `centers` and `neighs` hold global
                atom indices, so results index straight into the frame.

        Raises:
            ValueError: If any named species is absent, the cutoff string is not "Auto", or
                the selections resolve to more than one cutoff.
            TypeError: If the cutoff is of an unusable type.
            KeyError: If a cutoff dict has no entry for one of the bonds.
        """
        centers = _as_list(center_type)
        neighs = _as_list(neigh_type)
        if not centers or not neighs:
            raise ValueError("center_type and neigh_type must each name a species.")
        pairs = [(c, n) for c in centers for n in neighs]
        distinct = set(self._cutoffs(pairs, cutoff))
        if len(distinct) > 1:
            raise ValueError(
                f"A multi-species selection is measured at one cutoff, but {pairs} resolved "
                f"to {sorted(distinct)}. Call get_bonds once per bond, or pass a number."
            )
        (resolved,) = distinct
        return [frame.bonds(centers, neighs, resolved) for frame in self._frames(*centers, *neighs)]

    def get_angles(
        self, center_type: str, neigh_types: str | list[str], cutoff: Cutoff = "Auto", per_atom: bool = False
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Calculate the raw bond angles at each central atom, for every frame.

        The angle is measured between two neighbours of the central atom, so exactly two
        neighbour species define it: `neigh_types=["O", "Na"]` gives O-center-Na angles. A
        single string is taken to mean both arms of the angle.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (Union[str, List[str]]): The atomic symbol(s) of the neighbor atoms,
                either one symbol or a list of exactly two.
            cutoff (Cutoff, optional): Range within which to count a neighbour. Both arms
                of a same-species angle share one cutoff. Defaults to "Auto".
            per_atom (bool, optional): Group the angles by the central atom they were
                measured at, instead of pooling each frame's angles together. Defaults to
                False.

        Returns:
            Union[List[np.ndarray], List[List[np.ndarray]]]: By default, one array of
                angles in degrees per frame. With `per_atom=True`, one list per frame
                holding one array per central atom, in index order, empty for an atom with
                no qualifying neighbour pair.

        Raises:
            ValueError: If center_type or neigh_types are not present in the structure, or
                if neigh_types or an explicit cutoff list does not have exactly two entries.
        """
        arms = _angle_arms(neigh_types)
        cutoffs = self._cutoffs([(center_type, arm) for arm in arms], cutoff)
        per_frame = [_angles(frame, center_type, arms, cutoffs) for frame in self._frames(center_type, *arms)]
        if per_atom:
            return per_frame
        return [np.hstack(frame) if frame else np.array([], dtype=float) for frame in per_frame]

    def get_bridging_analysis(
        self,
        center_type: str,
        bridge_type: str,
        former_types: list[str] | None = None,
        cutoff: Cutoff = "Auto",
        per_atom: bool = False,
    ) -> dict[int, float] | list[np.ndarray]:
        """
        Calculate the Q^n speciation of a network former over the trajectory.

        A bridging atom is one of `bridge_type` that is bonded to at least two network
        formers, so `n` counts how many of a center atom's neighbours bridge on to another
        former. For a silicate, `get_bridging_analysis("Si", "O")` returns the fractions of
        Q0 through Q4.

        Two kinds of bond are measured: center-bridge, which sets `n`, and former-bridge,
        which decides whether a bridge atom bridges. Each former is judged at its own
        cutoff, so a mixed-former glass takes `cutoff={("Si", "O"): 1.7, ("B", "O"): 1.5}`.

        Args:
            center_type (str): The type of the center atoms.
            bridge_type (str): The type of the bridge atoms.
            former_types (Optional[List[str]], optional): The types counted as network
                formers when deciding whether a bridge atom bridges. Defaults to None,
                meaning `center_type` alone.
            cutoff (Cutoff, optional): The cutoff distance for considering a bond. Every
                bond here shares the bridge species as its neighbour, so telling the formers
                apart needs the pair-keyed form. Defaults to "Auto".
            per_atom (bool, optional): Return the raw per-atom counts for each frame instead
                of the aggregated distribution. Defaults to False.

        Returns:
            Union[Dict[int, float], List[np.ndarray]]: By default, a dictionary mapping each
                n to its fraction over the whole trajectory. With `per_atom=True`, one
                integer array per frame.

        Raises:
            ValueError: If center_type, bridge_type or any former_type is absent.
            TypeError: If former_types is neither None nor a list.
        """
        if former_types is not None and not isinstance(former_types, list):
            raise TypeError("former_types must be either None or a List of atom types")
        formers = list(dict.fromkeys(former_types if former_types else [center_type]))
        pairs = [(center_type, bridge_type)] + [(f, bridge_type) for f in formers]
        center_cutoff, *former_list = self._cutoffs(pairs, cutoff)
        former_cutoffs = dict(zip(formers, former_list))
        per_frame = [
            _bridging_analysis(frame, center_type, bridge_type, center_cutoff, former_cutoffs)
            for frame in self._frames(center_type, bridge_type, *formers)
        ]
        if per_atom:
            return per_frame
        return _fractions(np.concatenate(per_frame))

    def get_bridging_speciation(
        self,
        bridge_type: str,
        former_types: list[str] | str,
        cutoff: Cutoff = "Auto",
        per_atom: bool = False,
    ) -> dict[int, float] | list[np.ndarray]:
        """
        Calculate the speciation of a bridging atom over the trajectory.
        Counts how many network formers each `bridge_type` atom is bonded to. For a
        silicate, `get_bridging_speciation("O", "Si")` splits the oxygen into free oxygen
        at n=0, non-bridging oxygen at n=1, bridging oxygen at n=2 and tri-clusters at
        n=3 and above, so the whole classification comes out of one distribution rather
        than from labels fixed here.

        This is the same count that decides bridging in `get_bridging_analysis`, reported
        per bridge atom instead of reduced to a Q^n number per former.

        Args:
            bridge_type (str): The type of the bridging atoms, usually oxygen.
            former_types (Union[List[str], str]): The types counted as network formers.
            cutoff (Cutoff, optional): The cutoff distance for considering a bond. Every
                bond here shares the bridge species as its neighbour, so telling the formers
                apart needs the pair-keyed form. Defaults to "Auto".
            per_atom (bool, optional): Return the raw per-atom counts for each frame instead
                of the aggregated distribution. Defaults to False.

        Returns:
            Union[Dict[int, float], List[np.ndarray]]: By default, a dictionary mapping each
                number of formers to the fraction of bridge atoms bonded to that many, over
                the whole trajectory. With `per_atom=True`, one integer array per frame,
                aligned with the frame's `bridge_type` atoms in index order.

        Raises:
            ValueError: If bridge_type or any former_type is absent, or no former is named.
        """
        formers = list(dict.fromkeys(_as_list(former_types)))
        if not formers:
            raise ValueError("former_types must name at least one network former.")
        cutoffs = self._cutoffs([(f, bridge_type) for f in formers], cutoff)
        former_cutoffs = dict(zip(formers, cutoffs))
        per_frame = [
            _bridging_speciation(frame, bridge_type, former_cutoffs) for frame in self._frames(bridge_type, *formers)
        ]
        if per_atom:
            return per_frame
        return _fractions(np.concatenate(per_frame))

    def get_neighbors(self, center_type: str, cutoff: Cutoff = "Auto") -> list[dict[str, list[np.ndarray]]]:
        """
        Find the neighbors of each center atom of a given type, grouped by neighbor species.

        Args:
            center_type (str): The type of the center atoms.
            cutoff (Cutoff, optional): The cutoff distance for considering a neighbor.
                Every species is a neighbour here, so a dict needs an entry for each.
                Defaults to "Auto".

        Returns:
            List[Dict[str, List[np.ndarray]]]: One dict per frame, mapping each neighbor
                species to a list with one entry per center atom. Each entry is an array of
                global atom indices, so it indexes straight into the frame.

        Raises:
            ValueError: If center_type is not present in the structure.
            KeyError: If cutoff is a dict missing an entry for a species.
            TypeError: If cutoff is not a number, dict, or "Auto".
        """
        species = list(self.species)
        pairs = [(center_type, neigh_type) for neigh_type in species]
        cutoffs = dict(zip(species, self._cutoffs(pairs, cutoff)))
        return [_neighbors(frame, center_type, cutoffs) for frame in self._frames(center_type)]

    def get_angle_distribution(
        self,
        center_type: str,
        neigh_types: str | list[str],
        nbin: int = 70,
        cutoff: Cutoff = "Auto",
        range: tuple[float, float] | None = None,
        sin_normalised: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the angular distribution of a given pair of target atoms within a specified range.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_types (Union[str, List[str]]): The atomic symbols of the neighbor atoms.
            nbin (int, optional): The number of bins to use for the histogram. Defaults to 70.
            cutoff (Cutoff, optional): Range within which to calculate the angular
              distribution. Defaults to "Auto".
            range (Optional[Tuple[float, float]], optional): The range of the histogram.
              Defaults to None, meaning the full (0, 180) degrees an angle can take.
            sin_normalised (bool, optional): Normalise the distribution by sin(theta),
              removing the solid-angle weighting that favours angles near 90
              degrees even in an uncorrelated structure. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - angles: An array of shape (nbin,) containing the angle values (centers of bins).
                - dist: An array containing the angular distribution values (probability density).

        Raises:
            ValueError: If no angles were found at all, or none fell inside `range`, either
                of which would otherwise give a histogram of NaN from dividing a zero count
                by a zero total. Also if `sin_normalised` is asked for outside (0, 180),
                where the solid-angle weight is not defined.
        """
        arms = _angle_arms(neigh_types)
        pairs = [(center_type, arm) for arm in arms]
        cutoffs = self._cutoffs(pairs, cutoff)
        angles_all = np.hstack(self.get_angles(center_type, arms, dict(zip(pairs, cutoffs))))
        if range is None:
            range = (0.0, 180.0)
        if angles_all.size == 0:
            raise ValueError(
                f"No {arms[0]}-{center_type}-{arms[1]} angles found. The cutoff leaves the "
                f"central atoms with fewer than two neighbours to subtend an angle."
            )
        if sin_normalised and not (0.0 <= range[0] < range[1] <= 180.0):
            raise ValueError(
                f"sin_normalised needs range within (0, 180) degrees, got {range}: outside "
                f"it the sin(theta) weight is zero or negative and the density undefined."
            )

        counts, edges = np.histogram(angles_all, bins=nbin, range=range)
        widths = np.diff(edges)
        angles = 0.5 * (edges[:-1] + edges[1:])
        total = counts.sum()
        if total == 0:
            raise ValueError(f"All {angles_all.size} angles fell outside range={range}.")

        dist = counts.astype(float)
        if sin_normalised:
            solid = np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:]))
            dist = np.divide(dist, solid, out=np.zeros_like(dist), where=solid > 0)
        dist /= (dist * widths).sum()
        return angles, dist

    def get_coordination_numbers(
        self, center_type: str, neigh_type: str | list[str], cutoff: Cutoff = "Auto", per_atom: bool = False
    ) -> dict[int, float] | list[np.ndarray]:
        """
        Calculate the coordination number distribution over multiple frames.

        Args:
            center_type (str): The atomic symbol of the central atom.
            neigh_type (Union[str, List[str]]): The atomic symbol(s) of the
                neighbor atoms. Can be a single string (e.g. "O") or a list
                (e.g. ["O", "F"]) to count all neighbor types together.
            cutoff (Cutoff, optional): The cutoff distance. A dict tells the neighbour
                types apart. Defaults to "Auto".
            per_atom (bool, optional): Return the raw per-atom counts for each frame
                instead of the aggregated distribution. Defaults to False.

        Returns:
            Union[Dict[int, float], List[np.ndarray]]: By default, a dictionary mapping
                each coordination number to its fraction. With `per_atom=True`, one integer
                array per frame.

        Raises:
            ValueError: If center_type or any neigh_type is not found in the structure.
        """
        # Deduped: a species named twice would otherwise have its bonds counted twice.
        neigh_types = list(dict.fromkeys(_as_list(neigh_type)))
        if not neigh_types:
            raise ValueError("neigh_type must name at least one neighbour species.")
        cutoffs = self._cutoffs([(center_type, neigh) for neigh in neigh_types], cutoff)

        per_frame = [
            sum(frame.bonds(center_type, neigh, cut).counts() for neigh, cut in zip(neigh_types, cutoffs))
            for frame in self._frames(center_type, *neigh_types)
        ]
        if per_atom:
            return per_frame
        return _fractions(np.concatenate(per_frame))
