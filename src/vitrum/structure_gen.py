"""Unified glass composition and structure generation.

The :class:`GlassGenerator` class unifies several strategies for sampling
glass compositions and (optionally) realizing them as packed atomic structures
for glass simulations and materials discovery.

Two ways of describing a chemical system are supported:

* **Unit mode** -- pass ``units`` (a list of oxide/component formulas, e.g.
  ``["SiO2", "Na2O", "CaO"]``). Samples are mole fractions of each unit. Schemes:
  ``"sobol"``, ``"lhs"`` or ``"random"`` (stochastic simplex sampling) and
  ``"grid"`` (regular composition grid).
* **Elemental mode** -- pass ``elements``, a dict grouping element symbols by
  role, e.g. ``{"formers": ["Si"], "modifiers": ["Na"], "anions": ["O"]}``.
  Samples are charge-neutral atomic fractions. Scheme: ``"random"`` (random
  charge-balanced glasses).

Exactly one of ``units`` or ``elements`` must be given.

The workflow is always: build a generator, call :meth:`GlassGenerator.sample`
to get a :class:`Compositions` (a thin wrapper around a
:class:`pandas.DataFrame`), then optionally :meth:`Compositions.get_structures`
to pack the ones you want.

The module-level function :func:`gen_random_glasses` is retained as a thin,
backwards-compatible wrapper around the ``"random"`` scheme.
"""

import itertools
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy.stats.qmc import LatinHypercube, Sobol
from tqdm import tqdm

from vitrum.io_helpers import formula_unit
from vitrum.packing import get_random_packed

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ROUND = 4  # decimal places for mole-fraction output
_ZERO_THRESH = 1e-6  # values below this are treated as zero
_META_COLUMNS = ("n_components", "former_sum")
# Relative sampling weights per subsystem order for the "sobol" scheme
# (binary / ternary / quaternary).
_DEFAULT_ORDER_WEIGHTS = {2: 2, 3: 4, 4: 1}


@dataclass
class Compositions:
    """A set of sampled glass compositions produced by :meth:`GlassGenerator.sample`.

    Wraps the mole-fraction :class:`pandas.DataFrame` (accessible as ``.df``) and
    knows whether its component columns are oxide units (``mode="unit"``) or
    element symbols (``mode="elemental"``), which is all that is needed to convert
    the rows to pymatgen compositions, formula strings, or packed structures.

    Attributes:
        df (pandas.DataFrame): One column per component (mole fractions summing to
            1), plus a ``n_components`` column and, in unit mode, ``former_sum``.
        mode (str): ``"unit"`` or ``"elemental"``.
    """

    df: pd.DataFrame
    mode: str

    def __len__(self):
        return len(self.df)

    def _value_columns(self):
        return [c for c in self.df.columns if c not in _META_COLUMNS]

    @staticmethod
    def _units_to_composition(unit_fracs):
        comps = [Composition(u) * f for u, f in unit_fracs.items()]
        return sum(comps[1:], start=comps[0])

    def to_pymatgen(self):
        """Convert the sampled rows to pymatgen ``Composition`` objects.

        Each mole-fraction row is reduced to its smallest whole-number formula so
        the result is directly usable by :func:`vitrum.packing.get_random_packed`.

        Returns:
            list[pymatgen.core.Composition]
        """
        cols = self._value_columns()
        comps = []
        for _, row in self.df.iterrows():
            active = {c: row[c] for c in cols if row[c] > 0}
            comp = self._units_to_composition(active) if self.mode == "unit" else Composition(active)
            integer_formula, _ = comp.get_integer_formula_and_factor()
            comps.append(Composition(integer_formula))
        return comps

    def to_formulas(self, integers=True):
        """Convert the sampled rows to formula strings.

        Args:
            integers (bool): Return smallest whole-number formula units. Defaults to True.

        Returns:
            list[str]
        """
        cols = self._value_columns()
        formulas = []
        for _, row in self.df.iterrows():
            active = {c: row[c] for c in cols if row[c] > 0}
            if self.mode == "unit":
                mol_string = "-".join(f"{row[u]}{u}" for u in active)
                formulas.append(formula_unit(mol_string, integers=integers))
            else:
                comp = Composition(active)
                formulas.append(comp.get_integer_formula_and_factor()[0] if integers else comp.formula)
        return formulas

    def get_structures(self, target_atoms=100, datatype="ase", max_atoms=None, **packing_kwargs):
        """Realize the compositions as random-packed structures.

        Args:
            target_atoms (int): Target number of atoms per structure. Defaults to 100.
            datatype (str): ``"ase"`` or ``"pymatgen"``. Defaults to ``"ase"``.
            max_atoms (int, optional): Skip compositions whose packed cell would hold more
                than this many atoms. Checked before packing, which is the expensive step.
            **packing_kwargs: Forwarded to :func:`vitrum.packing.get_random_packed`.

        Returns:
            list: ASE ``Atoms`` or pymatgen ``Structure`` objects.
        """
        structures = []
        # Packing is the slow step, so it is the one worth showing progress for.
        for comp in tqdm(self.to_pymatgen()):
            if max_atoms is not None and _packed_cell_size(comp, target_atoms) > max_atoms:
                continue
            structures.append(get_random_packed(comp, target_atoms=target_atoms, datatype=datatype, **packing_kwargs))
        return structures


class GlassGenerator:
    """Generate glass compositions (and optionally structures) via several schemes.

    Provide *exactly one* of ``units`` (unit mode) or ``elements`` (elemental
    mode). There are no built-in chemistry defaults: passing neither, or both,
    raises a :class:`ValueError`.

    Args:
        units (list[str], optional): Oxide/component formulas for unit mode.
        elements (dict, optional): Element symbols grouped by role for elemental
            mode, with any of the keys ``"formers"``, ``"modifiers"`` and
            ``"anions"`` mapping to lists of symbols, e.g.
            ``{"formers": ["Si"], "modifiers": ["Na"], "anions": ["O"]}``.
        charges (dict, optional): Oxidation states used to charge-balance compositions
            in elemental mode. Any element not given here falls back to pymatgen's
            tabulated oxidation states.
        network_formers (set[str], optional): Which ``units`` count as network formers.
            This only labels the units — it applies no constraint on its own; the
            constraints are ``min_former_sum`` and the schemes' ``require_former``
            option. Defaults to none.
        x_min (float, optional): Minimum mole fraction of any active component in the
            stochastic unit-mode schemes (``"sobol"``, ``"lhs"``, ``"random"``).
            Defaults to 0.05.
        min_former_sum (float, optional): Minimum total network-former fraction in the
            unit-mode schemes. Defaults to 0.0, i.e. former-free compositions are
            allowed.
        seed (int, optional): Seed for reproducible sampling. Defaults to None.
    """

    _ELEMENT_GROUPS = ("formers", "modifiers", "anions")

    def __init__(
        self,
        units=None,
        elements=None,
        charges=None,
        network_formers=None,
        x_min=0.05,
        min_former_sum=0.0,
        seed=None,
    ):
        charges = charges or {}
        self.x_min = x_min
        self.min_former_sum = min_former_sum
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # Fixed base seed for the Sobol engines so runs are reproducible.
        self._sobol_base = int(self._rng.integers(0, 2**30))

        if (units is None) == (elements is None):
            raise ValueError(
                "Specify exactly one of units=[...] (unit mode) or "
                "elements={'formers': [...], 'modifiers': [...], 'anions': [...]} "
                "(elemental mode)."
            )

        if units is not None:
            self._mode = "unit"
            self.units = list(units)
            self.formers = self.modifiers = self.anions = []
            self._network_formers = set(network_formers) if network_formers is not None else set()
            self._elements = None
            self._charges = None
        else:
            self._mode = "elemental"
            unknown = set(elements) - set(self._ELEMENT_GROUPS)
            if unknown:
                raise ValueError(
                    f"Unknown element group(s) {sorted(unknown)}; expected any of {list(self._ELEMENT_GROUPS)}."
                )
            self.units = None
            self.formers = list(elements.get("formers", []))
            self.modifiers = list(elements.get("modifiers", []))
            self.anions = list(elements.get("anions", []))
            if not (self.formers or self.modifiers or self.anions):
                raise ValueError(f"elements must contain at least one of {list(self._ELEMENT_GROUPS)}.")
            self._elements = list(dict.fromkeys(self.modifiers + self.formers + self.anions))
            self._charges = self._resolve_charges(charges)

    # -- charge resolution ---------------------------------------------------

    def _resolve_charges(self, overrides):
        charges = {}

        def resolve(el, fallback_index):
            if el in overrides:
                return overrides[el]
            return Element(el).oxidation_states[fallback_index]

        for el in self.formers + self.modifiers:
            charges[el] = resolve(el, -1)  # most positive tabulated state
        for el in self.anions:
            charges[el] = resolve(el, 0)  # most negative tabulated state
        return charges

    # -- public API ----------------------------------------------------------

    #: Valid ``sample`` schemes per mode.
    _UNIT_SCHEMES = ("sobol", "lhs", "random", "grid")
    _ELEMENTAL_SCHEMES = ("random",)

    def sample(self, scheme="sobol", n=100, dedup=True, **scheme_kwargs):
        """Sample glass compositions using the given scheme.

        The valid schemes depend on the generator's mode:

        * **Unit mode**: ``"sobol"``, ``"lhs"``, ``"random"`` (stochastic simplex
          sampling) or ``"grid"`` (exhaustive regular grid).
        * **Elemental mode**: ``"random"`` (charge-balanced glasses).

        ``"random"`` is valid in both modes; the generator's mode selects which
        behaviour applies.

        Args:
            scheme (str): The sampling scheme (see above). Defaults to ``"sobol"``.
            n (int): Number of compositions to draw. Ignored by the exhaustive
                ``"grid"`` scheme.
            dedup (bool): Drop duplicate compositions. Defaults to True.
            **scheme_kwargs: Scheme-specific options. Unit-mode ``"sobol"``,
                ``"lhs"`` and ``"random"`` accept ``order_weights`` and
                ``require_former`` (whether every sample must contain at least one
                network former; defaults to True only when ``min_former_sum > 0``);
                ``"grid"`` accepts ``spacing``; the elemental ``"random"`` scheme
                accepts ``weights``.

        Returns:
            Compositions: A wrapper around the sampled mole-fraction DataFrame
            (available as ``.df``), with methods to convert the rows to pymatgen
            compositions, formula strings, or packed structures.
        """
        scheme = scheme.lower()
        if self._mode == "unit":
            if scheme not in self._UNIT_SCHEMES:
                raise ValueError(f"Unknown unit-mode scheme '{scheme}'. Choose from {list(self._UNIT_SCHEMES)}.")
            df = self._sample_continuous(n, scheme, dedup=dedup, **scheme_kwargs)
        else:
            if scheme not in self._ELEMENTAL_SCHEMES:
                raise ValueError(
                    f"Unknown elemental-mode scheme '{scheme}'. Choose from {list(self._ELEMENTAL_SCHEMES)}."
                )
            df = self._sample_random(n, weights=scheme_kwargs.get("weights", {}), dedup=dedup)

        df = df.reset_index(drop=True)
        value_cols = list(df.columns)
        df["n_components"] = (df[value_cols] > 0).sum(axis=1)
        if self._mode == "unit":
            former_cols = [c for c in value_cols if c in self._network_formers]
            df["former_sum"] = df[former_cols].sum(axis=1).round(_ROUND) if former_cols else 0.0
        return Compositions(df=df, mode=self._mode)

    # -- sampling schemes ----------------------------------------------------

    def _sample_continuous(self, n, scheme, dedup=True, order_weights=None, require_former=None, spacing=10):
        """Place points on the unit-composition simplex, then constrain and finalize.

        The raw points come from a scheme-specific generator (``"grid"`` enumerates
        a lattice; ``"sobol"`` / ``"lhs"`` / ``"random"`` draw points); the
        constraint and finishing steps (min_former_sum filter, rounding, dedup) are
        shared.
        """
        units = self.units
        former_idx = [i for i, u in enumerate(units) if u in self._network_formers]
        # Naming units as formers is not itself a constraint: only ask for a former
        # in every sample when min_former_sum actually demands one.
        if require_former is None:
            require_former = self.min_former_sum > 0

        if scheme == "grid":
            candidates = self._grid_candidates(spacing)
        else:
            candidates = self._stochastic_candidates(n, scheme, order_weights, require_former)

        rows, seen = [], set()
        for full in candidates:
            if former_idx and full[former_idx].sum() < self.min_former_sum:
                continue

            full = np.round(full, _ROUND)
            full[full < _ZERO_THRESH] = 0.0
            s = full.sum()
            if s == 0:
                continue
            full = np.round(full / s, _ROUND)

            if dedup:
                key = tuple(full)
                if key in seen:
                    continue
                seen.add(key)
            rows.append(full)

        if not rows:
            raise RuntimeError(
                "No compositions passed constraints. Try lowering min_former_sum or x_min, or a finer grid spacing."
            )
        return pd.DataFrame(rows, columns=units)

    def _grid_candidates(self, spacing):
        """Yield full unit-fraction vectors on a regular lattice summing to 1."""
        units = self.units
        axis = np.linspace(0, 100, int(100 / spacing + 1))
        for combo in itertools.product(axis, repeat=len(units)):
            if abs(sum(combo) - 100) > 1e-9:
                continue
            yield np.array(combo) / 100.0

    def _stochastic_candidates(self, n, scheme, order_weights, require_former):
        """Yield ``n`` full unit-fraction vectors drawn from subsystem simplices."""
        units = self.units
        n_dim = len(units)
        order_weights = order_weights or _DEFAULT_ORDER_WEIGHTS
        former_set = self._network_formers
        require_former = require_former and bool(former_set)

        subs_by_order = _enumerate_subsystems(units, order_weights, require_former, former_set)
        pool, weights = [], []
        for order, sub_list in subs_by_order.items():
            w = float(order_weights[order])
            for sub in sub_list:
                pool.append(sub)
                weights.append(w)
        if not pool:
            raise ValueError("No valid subsystems to sample. Check units, order weights, and require_former.")
        largest_order = max(len(sub) for sub in pool)
        if largest_order * self.x_min > 1.0:
            raise ValueError(
                f"x_min={self.x_min} cannot be satisfied for a {largest_order}-component "
                f"subsystem: {largest_order} components each need at least x_min, which "
                f"sums to {largest_order * self.x_min:.3f} > 1. Lower x_min to at most "
                f"{1.0 / largest_order:.3f}, or restrict order_weights to smaller subsystems."
            )
        weights = np.array(weights)
        weights /= weights.sum()

        chosen = self._rng.choice(len(pool), size=n, replace=True, p=weights)
        chosen_subs = [pool[i] for i in chosen]

        # Group draws by subsystem size so each engine draws a proper batch: LHS
        # (and Sobol) only fill space across the whole set, not point by point.
        by_size = defaultdict(list)
        for draw_i, sub in enumerate(chosen_subs):
            by_size[len(sub)].append(draw_i)

        simplex_points = [None] * n
        for size in sorted(by_size):
            positions = by_size[size]
            seed = int((self._sobol_base + size) % (2**30))
            unit_pts = _unit_hypercube(scheme, size - 1, len(positions), self._rng, seed)
            pts = _simplex_from_unit(unit_pts)
            for k, pos in enumerate(positions):
                simplex_points[pos] = pts[k]

        for draw_i, sub in enumerate(chosen_subs):
            comp_sub = simplex_points[draw_i] * (1.0 - len(sub) * self.x_min) + self.x_min

            full = np.zeros(n_dim)
            for j, gidx in enumerate(sub):
                full[gidx] = comp_sub[j]
            yield full

    def _sample_random(self, n, weights, dedup=True):
        rng = self._rng
        modifiers, formers, anions = self.modifiers, self.formers, self.anions
        charges = self._charges
        elements = self._elements

        num_mod_weights = weights.get("num_mod_weights", [0.5, 0.5, 0, 0])
        num_former_weights = weights.get("num_former_weights", [0.05, 0.65, 0.3, 0])
        num_anion_weights = weights.get("num_anion_weights", [0, 0.6, 0.4, 0])

        bias_modifiers = np.array(weights.get("bias_modifiers", [1] * len(modifiers)), dtype=float)
        bias_modifiers = bias_modifiers / bias_modifiers.sum() if len(modifiers) else bias_modifiers
        bias_formers = np.array(weights.get("bias_formers", [1] * len(formers)), dtype=float)
        bias_formers = bias_formers / bias_formers.sum() if len(formers) else bias_formers
        bias_anions = np.array(weights.get("bias_anions", [1] * len(anions)), dtype=float)
        bias_anions = bias_anions / bias_anions.sum() if len(anions) else bias_anions

        rows, seen = [], set()
        attempts, max_attempts = 0, n * 1000 + 1000
        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            num_mod = _choose_count(num_mod_weights, rng)
            num_former = _choose_count(num_former_weights, rng)
            num_anion = _choose_count(num_anion_weights, rng)

            if (num_mod + num_former) == 0 or num_anion == 0:
                continue
            if num_mod > len(modifiers) or num_former > len(formers) or num_anion > len(anions):
                continue

            chosen_mods = rng.choice(modifiers, num_mod, replace=False, p=bias_modifiers) if num_mod else []
            chosen_formers = rng.choice(formers, num_former, replace=False, p=bias_formers) if num_former else []
            chosen_anions = rng.choice(anions, num_anion, replace=False, p=bias_anions) if num_anion else []

            mod_form_ratio = (
                _random_partition(2, rng) if num_mod and num_former else [int(bool(num_mod)), int(bool(num_former))]
            )
            mod_ratio = _random_partition(num_mod, rng)
            form_ratio = _random_partition(num_former, rng)
            anion_ratio = _random_partition(num_anion, rng)

            avg_mod_charge = np.sum([charges[m] * mod_ratio[i] for i, m in enumerate(chosen_mods)])
            avg_form_charge = np.sum([charges[f] * form_ratio[i] for i, f in enumerate(chosen_formers)])
            avg_anion_charge = np.sum([charges[a] * anion_ratio[i] for i, a in enumerate(chosen_anions)])
            if avg_anion_charge == 0:
                continue

            avg_cation_charge = mod_form_ratio[0] * avg_mod_charge + mod_form_ratio[1] * avg_form_charge
            cation_anion_ratio = abs(avg_cation_charge) / abs(avg_anion_charge)

            amounts = []
            if num_mod:
                amounts.extend([_my_round(r * mod_form_ratio[0]) for r in mod_ratio])
            if num_former:
                amounts.extend([_my_round(r * mod_form_ratio[1]) for r in form_ratio])
            if num_anion:
                amounts.extend([_my_round(r * cation_anion_ratio) for r in anion_ratio])

            atoms = list(chosen_mods) + list(chosen_formers) + list(chosen_anions)
            int_amounts = np.rint(np.array(amounts) * 100).astype(int)
            total_charge = sum(a * charges[at] for a, at in zip(int_amounts, atoms))
            if total_charge != 0:
                balanced, final_charge = _balance_charge(int_amounts.tolist(), atoms, charges)
                if final_charge != 0:
                    continue
                int_amounts = np.array(balanced)

            if (int_amounts <= 0).any():
                continue

            formula = "".join(f"{at}{a}" for at, a in zip(atoms, int_amounts))
            if dedup and formula in seen:
                continue
            seen.add(formula)

            total = int_amounts.sum()
            row = {el: 0.0 for el in elements}
            for at, a in zip(atoms, int_amounts):
                row[at] += a / total
            rows.append(row)

        return pd.DataFrame(rows, columns=elements)


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper
# ---------------------------------------------------------------------------


def gen_random_glasses(
    modifiers, formers, anions, weights=None, num_structures=30, target_atoms=100, max_atoms=200, **kwargs
):
    """Generate random glass structures from given modifiers, formers and anions.

    Thin wrapper around ``GlassGenerator(...).sample("random")`` followed by
    packing. Kept for backwards compatibility; new code should use
    :class:`GlassGenerator` directly.

    Parameters:
        modifiers (list): Chemical symbols of the modifiers.
        formers (list): Chemical symbols of the network formers.
        anions (list): Chemical symbols of the anions.
        weights (dict, optional): Weights for the number of modifiers, formers and anions.
        num_structures (int, optional): Number of structures to generate. Defaults to 30.
        target_atoms (int, optional): Target number of atoms in each structure. Defaults to 100.
        max_atoms (int, optional): Skip compositions whose packed cell would exceed this many
            atoms. Defaults to 200.
        **kwargs: Additional keyword arguments passed to ``get_random_packed``.

    Returns:
        list: Random-packed structures (ASE ``Atoms`` or pymatgen ``Structure``).
    """
    generator = GlassGenerator(elements={"formers": formers, "modifiers": modifiers, "anions": anions})
    # `sample` already dedups and retries; oversample so the max_atoms filter, which is
    # applied after sampling, still has a chance of leaving num_structures behind.
    comps = generator.sample("random", n=num_structures * 3, weights=weights or {})
    structures = comps.get_structures(target_atoms=target_atoms, max_atoms=max_atoms, **kwargs)[:num_structures]

    if len(structures) < num_structures:
        warnings.warn(f"Only generated {len(structures)} of {num_structures} requested structures.")
    return structures


# ---------------------------------------------------------------------------
# Private sampling helpers
# ---------------------------------------------------------------------------


def _packed_cell_size(composition, target_atoms):
    """Number of atoms :func:`vitrum.packing.get_random_packed` would build for a composition.

    It packs whole formula units, so the cell holds the smallest multiple of the integer
    formula that reaches ``target_atoms``. Mirrors the sizing in ``get_random_packed`` so
    callers can filter on size without paying for the packing.
    """
    integer_composition = Composition(Composition(composition).get_integer_formula_and_factor()[0])
    return int(integer_composition.num_atoms * math.ceil(target_atoms / integer_composition.num_atoms))


def _my_round(x):
    """Round a value to the nearest 0.01, with 3 decimal places of precision."""
    return round(0.01 * round(x / 0.01), 3)


def _is_multiple(c, total, tol=1e-9):
    """Return True if ``total`` is (approximately) an integer multiple of ``c``."""
    ratio = total / c
    return math.isclose(ratio, round(ratio), abs_tol=tol)


def _choose_count(weights, rng):
    """Randomly choose an index (interpreted as a count) according to ``weights``."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return int(rng.choice(len(w), p=w))


def _random_partition(x, rng, total=10):
    """Randomly partition ``total`` into ``x`` positive parts, returned as fractions summing to 1."""
    if x <= 1:
        return np.array([1.0])
    cuts = np.sort(rng.choice(range(1, total), x - 1, replace=False))
    parts = np.diff([0] + list(cuts) + [total])
    return parts / total


def _balance_charge(amounts, atoms, charges_dict):
    """Adjust one atom's amount so the total ionic charge is closer to zero.

    Args:
        amounts (list[int]): Integer amounts of each atom type.
        atoms (list[str]): Chemical symbols corresponding to each entry in ``amounts``.
        charges_dict (dict): Mapping from chemical symbol to oxidation state.

    Returns:
        tuple[list[int], float]: The (possibly adjusted) amounts and the resulting total charge.
    """
    total_charge = sum(a * charges_dict[atom] for a, atom in zip(amounts, atoms))
    candidates = [(i, charges_dict[atom]) for i, atom in enumerate(atoms)]
    valid = [c for c in candidates if _is_multiple(c[1], total_charge)]
    if valid:
        idx, cand_charge = min(valid, key=lambda c: abs(abs(total_charge) - abs(c[1])))
        adjustment = int(round(total_charge / cand_charge))
        amounts[idx] += -adjustment
        new_charge = sum(a * charges_dict[atom] for a, atom in zip(amounts, atoms))
    else:
        new_charge = total_charge
    return amounts, new_charge


def _unit_hypercube(scheme, dim, m, rng, seed):
    """Draw ``m`` points in the ``[0, 1]^dim`` hypercube using the chosen engine.

    Args:
        scheme (str): One of ``"sobol"``, ``"lhs"`` or ``"random"``.
        dim (int): Dimensionality of the hypercube (may be 0).
        m (int): Number of points to draw.
        rng (numpy.random.Generator): RNG used for ``"random"``.
        seed (int): Seed for the quasi-random engines.

    Returns:
        numpy.ndarray: Array of shape ``(m, dim)``.
    """
    if dim == 0:
        return np.zeros((m, 0))
    if scheme == "random":
        return rng.random((m, dim))
    if scheme == "sobol":
        engine = Sobol(d=dim, scramble=True, seed=seed)
    elif scheme == "lhs":
        engine = LatinHypercube(d=dim, seed=seed)
    else:
        raise ValueError(f"Unknown sampling scheme '{scheme}'.")
    with warnings.catch_warnings():
        # Sobol warns when m is not a power of 2; sequence quality is adequate here.
        warnings.simplefilter("ignore")
        return engine.random(m)


def _simplex_from_unit(u):
    """Map rows of a ``[0, 1]^(d-1)`` sample onto the ``(d-1)``-simplex.

    Uses the order-statistics transform (sort, pad with 0 and 1, take successive
    differences). Returns an array of shape ``(m, d)`` whose rows sum to 1.
    """
    m = u.shape[0]
    if u.shape[1] == 0:
        return np.ones((m, 1))
    u_sorted = np.sort(u, axis=1)
    aug = np.concatenate([np.zeros((m, 1)), u_sorted, np.ones((m, 1))], axis=1)
    return np.diff(aug, axis=1)


def _enumerate_subsystems(units, order_weights, require_former, former_set):
    """Enumerate index-combinations of each order, optionally requiring a network former."""
    former_positions = {i for i, u in enumerate(units) if u in former_set}
    subs = {}
    for order in order_weights:
        if order > len(units):
            subs[order] = []
            continue
        valid = []
        for combo in itertools.combinations(range(len(units)), order):
            if not require_former or former_positions.intersection(combo):
                valid.append(list(combo))
        subs[order] = valid
    return subs
