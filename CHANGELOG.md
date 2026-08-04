# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.1.0] - 2026-08-04

### Fixed
- **The package did not import on a clean install.** `vitrum/__init__.py` reaches
  `volume_estimation` via `structure_gen` -> `packing`, and that module imported
  `atomate2` and `mp_api` at module scope even though both are optional extras. A fresh
  `pip install vitrum && python -c "import vitrum"` raised `ModuleNotFoundError`, and CI
  (which installs only `.[test]`) failed at collection. Both imports are now deferred into
  the branches that use them and report the missing extra by name.
- `Scattering.get_N_running` normalised by the density of the *centre* species rather than
  the *neighbour* species, so the two orderings of every cross pair were swapped.
  **Behaviour change**: results are wrong by the concentration ratio for any non-1:1
  stoichiometry -- exactly 4 instead of 8 for Ca-F in fluorite CaF2 -- and must be
  recomputed. 1:1 compositions are unaffected.
- `Scattering` captured `volume` and `aveden` from the first frame while the PDF backends
  correctly used per-frame values. They are now trajectory averages, which corrects
  `get_N_running`, `get_partial_structure_factor`, `get_T_r_pdf` and `get_reduced_pdf`
  for NPT trajectories. A trajectory whose composition changes between frames is now
  rejected rather than silently analysed with the first frame's species counts.
- `GlassAtoms.get_pdf` blanked the first histogram bin for *every* pair, discarding real
  cross-pair contacts that fell in it. Only like pairs have self-distances to remove.
- `Diffusion.get_diffusion_coef` raised a bare `ValueError("Inputs must not be empty.")`
  from `scipy` when `skip_first` (default 100) exceeded the trajectory length. It now
  reports how many frames the setting leaves. `Diffusion` also rejects a `sample_times`
  that does not match the trajectory length.
- `Scattering(use_neighborhood=True)` returned incorrect partial and total RDFs.
  Distances were keyed by a sorted element tuple but looked up with the unsorted
  pairs from `itertools.product`, so one ordering of every cross pair came back as
  exactly zero — on a two-species system the total RDF was ~19% too low, silently.
  **Any results produced with `use_neighborhood=True` should be recomputed.**
- `Scattering(use_neighborhood=True)` used the first frame's volume and chemical
  symbols for every frame; multi-frame trajectories with varying cell volume were
  normalised incorrectly.
- `find_rings(bonds=None)` raised `TypeError` despite the documented behaviour of
  allowing all bonds. The `bonds is not None` guard was missing from the
  bond-filtering loop.
- `homogeneity_checker` re-binned periodically wrapped atoms with a hardcoded index
  valid only for a 3x3x3 grid. Smaller grids raised `IndexError`; larger grids
  silently placed those atoms in the wrong box.
- `dimer_checker` ignored periodic boundaries, so dimers spanning a cell face were
  never detected.
- `Coordination.get_coordination_numbers` raised `UnboundLocalError` instead of a
  useful message when `cutoff` was an unrecognised string.

### Changed
- **Behaviour change**: the default `vol_per_atom_source` for `get_volume` and
  `get_random_packed` is now `"ionic_radius"` rather than `"mp"`, so the default path
  needs neither an optional extra nor a Materials Project API key. Volumes differ from the
  Materials Project estimate. The new estimator uses ionic radii and a packing fraction
  calibrated over 13 oxides (see `vitrum.volume_estimation.IONIC_PACKING_FRACTION`); it is accurate to
  roughly -20%/+25% in volume, i.e. under 10% in cell length. The calibration is
  oxide-centric -- for metallic or covalent systems, where no charge-balanced oxidation
  states exist and covalent radii are substituted, prefer an explicit `density=` or a float
  volume per atom.
- **Behaviour change**: partial PDFs for like pairs (e.g. Si-Si) are now normalised by
  `N_a * (N_a - 1)` rather than `N_a * N_a`, matching the standard definition. This applies
  to `GlassAtoms.get_pdf` as well as both `Scattering` backends; all three PDF
  implementations in the package now agree exactly. Affected results were low by
  `(N_a - 1) / N_a` — under 1% for a few hundred atoms, larger for small systems.
- `vitrum.geometry.pdf` is now the single normalisation primitive behind
  `GlassAtoms.get_pdf` and both `Scattering` backends, and takes an explicit `n_pairs`
  (the number of ordered pairs represented in the distances) and `exclude_self`. Both
  default to the previous behaviour for existing callers.
- Rings that wrap around the periodic cell are reported once, as a `warnings.warn` with a
  count, rather than one `print` per rejected ring. Non-convergence in `get_random_packed`
  and the two `VoidAnalysis` grid warnings are likewise `warnings.warn` now, so they can be
  filtered and captured.
- The ring search no longer raises a `SparseEfficiencyWarning` per bond (Guttman) or per
  atom (King, primitive); edges are dropped from the sparse graph's data array directly.
  Ring output is unchanged, verified bit-for-bit against the previous implementation.
- **Behaviour change**: `dimer_checker` no longer double-counts. The symmetric
  distance matrix was summed without halving, so every dimer counted twice; a given
  `num_allowed` threshold is now reached at half as many detections as before.
- `Scattering.get_total_rdf(type="xray")` now raises `NotImplementedError` instead of
  printing a message and returning an array of zeros.
- Triclinic cells now raise `NotImplementedError` via the new
  `vitrum.geometry.require_orthorhombic` helper, rather than silently producing wrong
  numbers from the orthorhombic-only minimum-image convention.
- Corrected return-type annotations and docstrings that disagreed with the code:
  `Scattering.get_weighted_partial_structure_factors` (returns a dict, not a tuple),
  `GlassAtoms.get_neighbors` (returns a dict, not a list), `correct_atom_types`
  (mutates in place, returns `None`) and `r_chi` (returns a 4-tuple, not a float).

### Added
- Medium-range-order analysis: `RingAnalysis`, `VoidAnalysis` and `PersistenceDiagram`.
- `GlassGenerator` and the packing/density utilities.
- A visualization module and formula utilities.
- `get_packing_radii` and `guess_oxi_states` are now public, and moved from `packing` to
  `volume_estimation`, which needs them for the `"ionic_radius"` estimator. They could not
  stay in `packing`, which imports `volume_estimation`, without making the dependency
  circular. They were previously private as `_get_packing_radii` / `_guess_oxi_states`.
- `vitrum.geometry.radial_bins`, replacing four copies of the bin-edge and shell-volume
  computation.
- Tests for the clean-install import path, the running coordination number against
  fluorite CaF2, agreement between `GlassAtoms.get_pdf` and `Scattering`, trajectory
  averaging of the number density, and the `Diffusion` input guards.
- A `pytest` suite under `tests/`, covering the analysis modules against analytic
  reference structures (crystalline silicon's known coordination, bond angle and ring
  statistics; ideal-gas g(r) and S(Q) asymptotics). Install with `pip install -e .[test]`.
- A GitHub Actions test workflow running the suite on Python 3.10-3.13.

## [1.0.1]

### Added
- Zenodo archiving is now linked to GitHub releases, giving each release a
  citable DOI. Added `CITATION.cff` and a citation section to the README and
  docs.

### Changed
- Updated installation docs to reference the PyPI package.

## [1.0.0]

First stable release, published to PyPI.

### Changed
- Migrated packaging from `setup.py` to `pyproject.toml` (PEP 621); version is now
  sourced from `vitrum.__version__` as the single source of truth.
- Raised the minimum supported Python version to 3.10.
- Split the former `vitrum.utility` grab-bag module into focused modules:
  `geometry`, `trajectory`, `packing`, `volume_estimation`, `io_helpers`,
  `structure_validation`, `comparison`.
- Renamed `glass_Atoms.py` to `glass_atoms.py` and removed the redundant
  lowercase class aliases (`glass_Atoms`, `coordination`, `diffusion`,
  `scattering`, `RINGs`) in favor of their PascalCase names
  (`GlassAtoms`, `Coordination`, `Diffusion`, `Scattering`, `RingAnalysis`).
- `vitrum/__init__.py` now exports the five main classes directly
  (`from vitrum import GlassAtoms`, etc.).


 