# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-08-05

This release corrects several bugs that produced silently wrong numbers.
**Ring statistics and topology metrics, running coordination numbers, partial PDFs for like
pairs, broadened RDFs, anything computed with `Scattering(use_neighborhood=True)` should be
recomputed.**

### Added

- **`Scattering.get_total_rdf(type="xray")` is implemented** and no longer raises
  `NotImplementedError`, which `get_T_r_pdf` and `get_reduced_pdf` inherit. Follows Keen
  (2001) §5, with the transform truncated at the instance's `qmax`.
- `lorch=True` on `get_total_rdf`, `get_T_r_pdf` and `get_reduced_pdf`, suppressing the
  truncation ripple in the X-ray RDF. It raises for the other weightings, which involve no
  transform. See the scattering docs on choosing `qmax`.
- `get_weighted_partial_structure_factors` accepts `type="approx_xray"`, which previously
  raised.
- Medium-range-order analysis: `RingAnalysis`, `VoidAnalysis` and `PersistenceDiagram`, with
  ring topology metrics on `Ring` — `area()`, `eccentricity()`, `planeness()` and
  `ellipse_eccentricity()`.
- `GlassGenerator` and the packing/density utilities, plus a visualization module and formula
  utilities.
- `vitrum.bonds.Bonds` and `Coordination.get_bonds`: the raw bond list behind coordination
  numbers, neighbour lists, angles and Q^n speciation, so analyses the package does not ship
  can be derived without reaching into internals. `counts()`, `degrees()`, `lists()`,
  `lengths(atoms)` and `select_neighs()`.
- `Coordination.get_bridging_analysis`, `get_angles` and `get_neighbors`, the trajectory-aware
  replacements for the corresponding `GlassAtoms` methods.
- `Coordination.get_bridging_speciation`, the oxygen speciation: how many network formers each
  bridging atom is bonded to. Free oxygen at `n=0`, non-bridging at `n=1`, bridging at `n=2`,
  tri-clusters above.
- `per_atom=True` on `get_coordination_numbers`, `get_bridging_analysis` and `get_angles`: one
  entry per frame holding raw per-centre-atom values instead of the trajectory-wide summary.
- **Per-bond cutoffs, `cutoff={("Si", "O"): 1.6, ("B", "O"): 1.4}`**, in every method that
  takes a cutoff. `{"O": 1.6}` remains as shorthand, key order is irrelevant, and `"Auto"`
  resolves each bond separately — so `get_bridging_analysis` judges each network former at its
  own cutoff rather than measuring B-O against the Si-O bond length.
- `find_rings` and `RingAnalysis.calculate` accept `cutoff`, building the bond graph from a
  distance instead of covalent radii. `cutoff=None` (the default) keeps the previous behaviour.
- `sin_normalised=True` on `get_angle_distribution`, for the P(theta)/sin(theta) convention.
- `vitrum.geometry.peak_metrics`: position, FWHM and height of the first peak of a tabulated
  function — bond length from a partial g(r), first sharp diffraction peak from S(Q). An
  optional `window` restricts the search.
- `vitrum.geometry.distance_matrix`, `vitrum.geometry.partial_pdf` and
  `vitrum.io_helpers.get_density`.
- A `pytest` suite under `tests/`, checked against independent references. Install with
  `pip install -e .[test]`; CI runs it on Python 3.10-3.13.

### Changed

Each of the following changes the numbers the package returns.

- **The default `vol_per_atom_source` for `get_volume` and `get_random_packed` is now
  `"ionic_radius"`, not `"mp"`**, so the default path needs no Materials Project API key.
- **Partial PDFs for like pairs (e.g. Si-Si) are normalised by `N_a * (N_a - 1)`** rather than
  `N_a * N_a`. `GlassAtoms.get_pdf` and both `Scattering` backends now agree exactly.
- **`find_rings(limit=...)` / `RingAnalysis.calculate(max_size=...)` now mean the number of
  atoms in the ring**, and mean the same thing for every criterion; they previously capped
  path hops.
- **`criterion="primitive"` repeats the cell 3x3x3 by default** on a periodic structure. Pass
  `repeat` explicitly to override.
- **`cutoff="Auto"` is resolved once from a single frame in every `Coordination` method**,
  rather than per frame. `Coordination(frames, cutoff_frame=-1)` picks the frame.
- **`Coordination.get_neighbors` returns global atom indices** rather than indices into each
  species' own atoms.
- **`Coordination.get_angles` returns one flat array of angles per frame.** The per-centre
  grouping is now `get_angles(..., per_atom=True)`, which keeps an empty entry for centres
  with no qualifying pair.
- **`get_coordination_numbers` no longer double-counts a repeated neighbour species.**
  `neigh_type=["O", "O"]` reported every coordination number doubled.
- `Coordination`, `vitrum.bonds` and cutoff-based ring searches accept a triclinic cell. 
- `Scattering`, `vitrum.geometry`, `vitrum.voids` and trajectory unwrapping raise
  `NotImplementedError` on a triclinic cell rather than returning wrong numbers.
  `Scattering.get_total_rdf(type="xray")` likewise raises instead of returning zeros.
- Cutoffs are validated through one shared resolver: only the exact string `"Auto"` is
  accepted, an unusable type raises `TypeError`, an unusable value `ValueError`, and a dict
  with no entry for a bond `KeyError`.
- Failures that surfaced as `UnboundLocalError`, `IndexError` or bare scipy `ValueError`s now
  raise messages naming the problem.

### Removed

- **A cutoff can no longer be given as a bare list** such as `cutoff=[1.6, 2.4]`; pass a number
  or a dict keyed by bond or by neighbour species. It follows that the two arms of a
  same-species angle always share one cutoff.
- `vitrum.io_helpers.parse_composition` (use `pymatgen.core.Composition` directly) and
  `vitrum.mlip_functions.min_max_val`.
- `Diffusion.get_van_hove_dist_correlation` and `Diffusion.get_velocity_autocorrelation`, which
  were empty stubs.
- The `scikit-learn` and `ruamel-yaml` dependencies. `vitrum.batch_active` still needs
  scikit-learn, which the `workflows` extra declares.

### Deprecated

- **`GlassAtoms` is deprecated and will be removed in 2.0.0.** Every analysis class now takes
  plain `ase.Atoms`. Existing code keeps working: each method returns what it returned before
  and warns with the name of its replacement. **This changes no numbers.**

  | Deprecated | Use instead |
  | --- | --- |
  | `GlassAtoms(atoms).get_dist()` | `vitrum.geometry.distance_matrix(atoms)` |
  | `GlassAtoms(atoms).get_pdf(pair)` | `Scattering(atoms).get_partial_pdf(pair)` |
  | `GlassAtoms(atoms).get_all_angles(...)` | `Coordination([atoms]).get_angles(...)` |
  | `GlassAtoms(atoms).get_coordination_number(...)` | `Coordination([atoms]).get_coordination_numbers(..., per_atom=True)` |
  | `GlassAtoms(atoms).get_bridging_analysis(...)` | `Coordination([atoms]).get_bridging_analysis(...)` |
  | `GlassAtoms(atoms).get_neighbors(...)` | `Coordination([atoms]).get_neighbors(...)` |
  | `GlassAtoms(atoms).get_density()` | `vitrum.io_helpers.get_density(atoms)` |
  | `GlassAtoms(atoms).set_new_chemical_symbols(map)` | `vitrum.io_helpers.correct_atom_types(atoms, map)` |

  See [Migrating from GlassAtoms](docs/vitrum/glass_atoms.md). The deprecated methods keep the
  conventions they shipped with, so `GlassAtoms.get_neighbors` still returns species-relative
  indices, `get_all_angles` still groups angles by centre atom, and `get_bridging_analysis`
  still applies a single cutoff to every former-bridge bond.

### Fixed

**Ring analysis** — every ring count and size distribution produced before this release
changes.

- All three criteria are defined over every shortest path, but the implementations kept only
  one per search. Replaced with a BFS enumeration of all shortest paths, which is also faster.
- `criterion="primitive"` filtered King's ring candidates and so missed most primitive rings.
  Replaced with the `PRIM_DIJKSTRA` enumeration from R.I.N.G.S.
- A ring wrapping the periodic cell ended the search for its bond, returning nothing rather
  than the next-shortest real ring.
- Ring topology metrics were wrong for any ring spanning more than half the cell. Rings are now
  unwrapped bond by bond along the ring, which holds at any ring size.
- Atom pairs bonded through more than one periodic image, and rings folded onto their own
  periodic image, are discarded with a warning to increase `repeat`.
- `find_rings(bonds=None)` raised `TypeError` despite the documented behaviour, and an empty
  bond graph raised a scipy `ValueError` instead of returning no rings.

**Scattering and PDFs**

- The minimum image convention subtracted at most one cell length. Distances no longer depend on whether the input was wrapped.
- `Scattering(use_neighborhood=True)` returned incorrect partial and total RDFs: one ordering
  of every cross pair came back as exactly zero.
- `Scattering.get_N_running` was normalised by the density of the centre species rather than the
  neighbour species.
- `Scattering` took `volume`, `aveden`, chemical symbols, the orthorhombic check and the
  default `rrange` from the first frame, so NPT trajectories were normalised incorrectly. These
  are now trajectory averages, every frame is checked, and a trajectory whose composition
  changes between frames is rejected.
- `gaussian_broadening` applied the truncation kernel to `g(r)` rather than to `r·g(r)`, so a
  coordination shell's area grew with the broadening. It is now conserved.
- `unwrap_trajectory` used the first frame's cell for the whole trajectory, so an atom at fixed
  fractional coordinates through an NPT cell rescaling registered as having diffused.
- `Diffusion` rejected ordinary wrapped trajectories: the guard against double-unwrapping
  required every coordinate to lie strictly inside the cell.

**Structure generation and validation**

- `get_volume` silently ignored `density` when `vol_per_atom_source` was also a float. Passing
  both now raises, as does `vol_per_atom_source="density"` with no density.
- `get_random_packed` reseeded the global NumPy RNG; it now uses a local `Generator`.
- `homogeneity_checker` used an index hardcoded for a 3x3x3 grid, and `dimer_checker` ignored
  periodic boundaries. Both also wrapped the caller's atoms in place, and now copy first.

**Packaging** — the package did not import on a clean install: `structure_gen` → `packing`
imported the optional `atomate2` and `mp_api` at module scope. Both are now deferred and report
the missing extra by name. `tqdm` and `numpy>=2.0` are declared as dependencies.


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