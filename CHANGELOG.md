# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-08-05

This release corrects several bugs that produced silently wrong numbers.
**Ring statistics, running coordination numbers, partial PDFs for like pairs, and
anything computed with `Scattering(use_neighborhood=True)` or on an NPT trajectory
should be recomputed.**

### Added

- Medium-range-order analysis: `RingAnalysis`, `VoidAnalysis` and `PersistenceDiagram`.
- `GlassGenerator` and the packing/density utilities, plus a visualization module and
  formula utilities.
- Ring topology metrics on `Ring`, following the analysis in Supplemental note 1:
  `area()`, `eccentricity()`, `planeness()` and `ellipse_eccentricity()`. All are computed
  from PBC-unwrapped positions, so a ring straddling a cell face measures the same as one
  inside it. `ellipse_eccentricity()` returns NaN below five atoms, which is how many
  points are needed to fix a conic.
- A `pytest` suite under `tests/`, checked against analytic reference structures
  (crystalline silicon's coordination, bond angles and ring statistics; ideal-gas g(r) and
  S(Q) asymptotics). Install with `pip install -e .[test]`. A GitHub Actions workflow runs
  it on Python 3.10–3.13.

### Changed

Each of the following changes the numbers the package returns.

- **The default `vol_per_atom_source` for `get_volume` and `get_random_packed` is now
  `"ionic_radius"`, not `"mp"`**, so the default path needs neither an optional extra nor a
  Materials Project API key. Volumes differ from the Materials Project estimate.
- **Partial PDFs for like pairs (e.g. Si-Si) are now normalised by `N_a * (N_a - 1)`**
  rather than `N_a * N_a`, matching the standard definition. This applies to
  `GlassAtoms.get_pdf` and both `Scattering` backends, which now agree exactly. Affected
  results were low by `(N_a - 1) / N_a` — under 1% for a few hundred atoms, larger for
  small systems.
- **`find_rings(limit=...)` / `RingAnalysis.calculate(max_size=...)` now mean the number of
  atoms in the ring**, and mean the same thing for every criterion. The value was
  previously passed straight to `dijkstra`, which caps path *hops*, so it admitted rings
  one atom larger for `guttman` and two larger for `king`/`primitive` — `max_size=4` still
  returned 6-rings. A finite `limit` below 3 is now rejected.
- **`criterion="primitive"` now repeats the cell 3×3×3 by default** on a periodic
  structure, because a single minimum-image cell cannot represent every primitive ring.
  R.I.N.G.S. replicates unconditionally for this criterion. Pass `repeat` explicitly to
  override; it defaults to `None` rather than `(1, 1, 1)`.
- **`dimer_checker` no longer double-counts.** The symmetric distance matrix was summed
  without halving, so a given `num_allowed` threshold is now reached at half as many
  detections as before.
- Triclinic cells now raise `NotImplementedError` rather than silently producing wrong
  numbers from the orthorhombic-only minimum-image convention.
- `Scattering.get_total_rdf(type="xray")` raises `NotImplementedError` instead of printing
  a message and returning an array of zeros.
- Rejected wrapping rings, `get_random_packed` non-convergence and the `VoidAnalysis` grid
  problems are reported once through `warnings.warn` rather than per-item `print`, so they
  can be filtered and captured.

### Fixed

**Ring analysis** — every ring count and size distribution produced before this release
changes.

- Ring criteria kept only one shortest path per search. All three criteria are defined over
  *every* shortest path, but the implementations took a single `dijkstra` predecessor
  chain, so equally short paths closing the same bond were dropped. Against diamond
  silicon's known 12 six-rings per atom, `guttman` returned 98 and `king`/`primitive` 124
  for the 64-atom cell; all three now return the correct 128, and 432 for 216 atoms. The
  replacement enumerates all shortest paths by BFS and is faster.
- `criterion="primitive"` missed most primitive rings. It filtered King's ring candidates,
  but primitive rings are not a subset of King's rings. On a 27-atom simple-cubic cell it
  returned 81 of the 189 primitive rings; on the cube graph, 6 of 10. Replaced with the
  `PRIM_DIJKSTRA` enumeration from R.I.N.G.S.
- A ring wrapping the periodic cell ended the search for its bond, returning nothing rather
  than the next-shortest real ring — for a single cubic silicon cell `guttman` returned
  **no rings at all**, and now returns diamond's correct 16 six-rings.
- In small cells, atom pairs bonded through more than one periodic image kept only the last
  offset, so rings were compared against an arbitrary one and mis-classified as periodic or
  not. Extra images are now skipped with a warning to increase `repeat`. Self-image bonds,
  which corrupted the offset map and produced spurious one-atom candidates, are likewise
  skipped.
- `find_rings(bonds=None)` raised `TypeError` despite the documented behaviour of allowing
  all bonds; an empty bond graph raised a scipy `ValueError` instead of returning no rings.

**Scattering and PDFs**

- `Scattering.get_N_running` normalised by the density of the *centre* species rather than
  the *neighbour* species, swapping the two orderings of every cross pair.
- `Scattering(use_neighborhood=True)` returned incorrect partial and total RDFs. Distances
  were keyed by a sorted element tuple but looked up with unsorted pairs, so one ordering
  of every cross pair came back as exactly zero; on a two-species system the total RDF was
  ~19% too low, silently.
- `Scattering` captured `volume`, `aveden` and chemical symbols from the first frame, so
  NPT trajectories were normalised incorrectly throughout `get_N_running`,
  `get_partial_structure_factor`, `get_T_r_pdf` and `get_reduced_pdf`. These are now
  trajectory averages, and a trajectory whose composition changes between frames is
  rejected rather than analysed with the first frame's counts.
- `GlassAtoms.get_pdf` blanked the first histogram bin for every pair, discarding real
  cross-pair contacts; only like pairs have self-distances to remove.

**Structure generation and validation**

- `get_volume` silently ignored `density` when `vol_per_atom_source` was also a float,
  sizing the cell from the wrong quantity (480 Å³ rather than 363 Å³ for SiO₂ at
  2.2 g/cm³). Passing both now raises, as does `vol_per_atom_source="density"` with no
  density.
- `GlassGenerator` did not honour `x_min`: fractions were clipped to the floor and then
  renormalized back below it, leaving 111 of 1299 nonzero fractions under a requested
  `x_min=0.05`. Sampling now maps onto the reduced simplex. An `x_min` too large for the
  largest subsystem is rejected up front.
- Charge balancing in the elemental `"random"` scheme truncated binary-rounded floats
  (0.29 became 28 rather than 29), unbalancing the charge sum.
- `get_random_packed` reseeded the **global** NumPy RNG via `np.random.seed`, destroying
  the caller's reproducibility; it now uses a local `Generator`.
- `homogeneity_checker` re-binned wrapped atoms with an index hardcoded for a 3×3×3 grid,
  raising `IndexError` on smaller grids and misplacing atoms on larger ones.
  `dimer_checker` ignored periodic boundaries, so dimers spanning a cell face were never
  detected. Both also called `atoms.wrap()` on the caller's object, moving their atoms as a
  side effect of an inspection call, and now copy first.

**Packaging**

- The package did not import on a clean install. `vitrum/__init__.py` reaches
  `volume_estimation` via `structure_gen` → `packing`, which imported the optional
  `atomate2` and `mp_api` at module scope, so `pip install vitrum && python -c "import
  vitrum"` raised `ModuleNotFoundError` and CI failed at collection. Both imports are now
  deferred and report the missing extra by name.
- Declared `numpy>=2.0`. `vitrum.scattering` uses `np.trapezoid`, added in NumPy 2.0, so
  installs resolving NumPy 1.x failed with `AttributeError` at call time.

**Error reporting** — failures that surfaced as `UnboundLocalError`, `IndexError` or bare
scipy `ValueError`s now raise messages that name the problem: an unrecognised `cutoff`
string or `algorithm`, a PDF with no local minimum after its first peak (the failure mode
of every `cutoff="Auto"` path), a `skip_first` longer than the trajectory, a mismatched
`sample_times`, and species absent from the structure. Return-type annotations and
docstrings that disagreed with the code were corrected for
`Scattering.get_weighted_partial_structure_factors`, `GlassAtoms.get_neighbors`,
`correct_atom_types` and `r_chi`.


## [1.0.1]

### Added

- Zenodo archiving linked to GitHub releases, giving each release a citable DOI, plus
  `CITATION.cff` and a citation section in the README and docs.

### Changed

- Updated installation docs to reference the PyPI package.

## [1.0.0]

First stable release, published to PyPI.

### Changed

- Migrated packaging from `setup.py` to `pyproject.toml` (PEP 621); version is sourced from
  `vitrum.__version__` as the single source of truth.
- Raised the minimum supported Python version to 3.10.
- Split the former `vitrum.utility` grab-bag module into `geometry`, `trajectory`,
  `packing`, `volume_estimation`, `io_helpers`, `structure_validation` and `comparison`.
- Renamed `glass_Atoms.py` to `glass_atoms.py` and removed the redundant lowercase class
  aliases in favor of `GlassAtoms`, `Coordination`, `Diffusion`, `Scattering` and
  `RingAnalysis`.
- `vitrum/__init__.py` now exports the five main classes directly.