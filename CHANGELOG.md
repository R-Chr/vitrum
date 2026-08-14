# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-08-13

This release corrects several bugs that produced silently wrong numbers.
**Ring statistics and topology metrics, running coordination numbers, partial PDFs for like
pairs, broadened RDFs, and anything computed with 1.0.1's `Scattering(use_neighborhood=True)`
should be recomputed.**

### Added

- **`Scattering.get_total_rdf(type="xray")` is implemented** and no longer raises
  `NotImplementedError`, which `get_T_r_pdf` and `get_reduced_pdf` inherit. Follows Keen
  (2001) §5, with the transform truncated at the instance's `qmax`.
- `lorch=True` on `get_total_rdf`, `get_T_r_pdf` and `get_reduced_pdf`, suppressing the
  truncation ripple in the X-ray RDF. It raises for the other weightings, which involve no
  transform. See the scattering docs on choosing `qmax`.
- `get_weighted_partial_structure_factors` accepts `type="approx_xray"`, which previously
  raised.
- **`vitrum.voids.VoidAnalysis`**: cavities from an occupancy grid, their volumes and radii,
  export as pseudo-atoms, and a 3D isosurface plot.
- **`vitrum.persistent_homology.PersistenceDiagram`**, replacing the broken `LocalPD` and the
  loose functions around it. Diagrams for dimensions 1 and 2, the atoms making up a cycle,
  diagram composition, accumulated persistence, size-persistence histograms and persistence
  images, each with a plot method.
- Ring topology metrics on `Ring` — `area()`, `eccentricity()`, `planeness()` and
  `ellipse_eccentricity()`.
- **`GlassGenerator` and `Compositions`**, for sampling glass compositions across a system —
  on a grid, at random or stochastically, charge-balanced, with the subsystems enumerated —
  and turning them into packed structures via `get_structures`.
- **`vitrum.visualization.StructureRenderer`**, rendering `ase.Atoms` to static images or
  Jupyter widgets through OVITO. Needs the `visualization` extra.
- Density and formula utilities: `vitrum.io_helpers.get_density`,
  `mass_density_to_number_density`, `number_density_to_mass_density` and `formula_unit`, plus
  `vitrum.volume_estimation.get_packing_radii` and `guess_oxi_states`, the per-atom radii
  behind the hard-sphere overlap resolution in `packing`.
- `Bonds`, `Compositions`, `GlassGenerator` and `VoidAnalysis` are exported from `vitrum`
  directly, alongside the classes that were already there.
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
- `vitrum.geometry.distance_matrix` and `vitrum.geometry.partial_pdf`.
- A `pytest` suite under `tests/`, checked against independent references. Install with
  `pip install -e .[test]`; CI runs it on Python 3.10-3.13.
- **A `py.typed` marker (PEP 561)**, so the type annotations throughout the package are
  visible to `mypy`, `pyright` and editors in downstream projects. They were silently ignored
  before, because an installed distribution without the marker is treated as untyped.
- **The public API is now fully annotated**, and CI enforces it with `mypy` under
  `disallow_untyped_defs` and `disallow_incomplete_defs` — the marker above would otherwise
  promise type information the package did not have. Every function outside the unsupported
  `vitrum.batch_active` carries argument and return annotations.
- Community documentation: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and issue
  and pull-request templates.
- Docs for `vitrum.visualization`, the one module whose API reference was not reachable from
  the documentation nav.
- **Triclinic cells in `vitrum.geometry.distance_matrix`**, and so in `Scattering` and
  `GlassAtoms.get_pdf`. The cell is reduced to a Minkowski basis and the nearest periodic image
  searched for; a cell with perpendicular axes needs no search and costs what it always did.
  Verified against `ase`'s own minimum-image distances. `vitrum.geometry.get_dist`.
- `vitrum.geometry.perpendicular_widths`: the distance between each pair of opposite cell
  faces, which is what bounds the minimum image convention. Equal to the cell lengths for an
  orthorhombic cell, shorter than them for a sheared one.
- **`vitrum.geometry.cell_list_pair_counts`, the cell-list kernel now backing `Scattering`'s
  PDFs.** Distances are histogrammed as they are found, so memory is O(N) rather than one
  record per pair: 10⁶ atoms at `rrange` = 20 Å bins 1.7 × 10⁹ pairs in seconds, which
  previously did not fit in memory. It is also 8-20× faster than the distance matrix on a
  3,000-atom cell, at identical results. It needs a cell periodic along all three axes and
  `rrange` within the minimum image limit, and raises otherwise.
- `vitrum.geometry.minimum_image_limit`: the largest radius at which the minimum image
  convention holds, and the bound `Scattering` and `cell_list_pair_counts` enforce `rrange`
  against. It is the narrower of the cell as given and its Minkowski reduction, since neither
  is reliably the tighter of the two.
- **`vitrum[fast]` extra**, installing [matscipy](https://github.com/libAtoms/matscipy). The
  neighbour search under `Coordination`, `Bonds` and the partial PDFs is where nearly all of
  their runtime goes; matscipy's C++ implementation replaces the ASE one when it is present.

### Changed

Each of the following changes the numbers the package returns.

- **`Scattering` rejects an `rrange` above half the shortest perpendicular cell width**, where
  it previously warned and returned numbers anyway. Run a larger cell to tabulate further.
- **`Scattering`'s default `rrange` is capped at 20 Å**. Cells up to 40 Å across are
  unaffected; above that the default no longer tracks the cell.
- **The default `vol_per_atom_source` for `get_volume` and `get_random_packed` is now
  `"ionic_radius"`, not `"mp"`**, so the default path needs no Materials Project API key.
- **Partial PDFs for like pairs (e.g. Si-Si) are normalised by `N_a * (N_a - 1)`** rather than
  `N_a * N_a`. `GlassAtoms.get_pdf` and `Scattering` now agree exactly.
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
- `Coordination`, `vitrum.bonds` and cutoff-based ring searches accept a triclinic cell, as
  does `Scattering` through the distance and cell-list kernels above.
- **`vitrum.trajectory` is now `vitrum.trajectory_tools`**, so that the module is not confused
  with a trajectory object. `unwrap_trajectory` and `get_high_low_displacement_index` are
  unchanged; only the import path moves.
- The routines that still apply the minimum image convention with the cell lengths alone —
  `trajectory_tools.unwrap_trajectory`, `voids.build_void_grid` and `VoidAnalysis`, and
  `structure_validation.homogeneity_checker` — raise `NotImplementedError` on a triclinic cell
  rather than returning wrong numbers, through the shared
  `vitrum.geometry.require_orthorhombic` guard. See [Known
  issues](docs/vitrum/known_issues.md).
- Cutoffs are validated through one shared resolver: only the exact string `"Auto"` is
  accepted, an unusable type raises `TypeError`, an unusable value `ValueError`, and a dict
  with no entry for a bond `KeyError`.
- Failures that surfaced as `UnboundLocalError`, `IndexError` or bare scipy `ValueError`s now
  raise messages naming the problem.

### Removed

- **`Scattering(use_neighborhood=...)`.** There is one backend now, so there is nothing to
  select; passing the argument raises `TypeError`. It only ever traded speed, and the cell list
  is the faster of the two.
- **`Scattering.calculate_partial_pdfs_neighborhood` is now
  `Scattering.calculate_partial_pdfs_cell_list`**, which is what it does; the
  `ase.neighborlist` path it was named for is gone.
- **`Scattering.calculate_partial_pdfs` is now private, `_partial_pdfs_dense`.** It stopped
  being the backend in this release and was only a slower route to the same numbers, with no
  effect unless its result was assigned over `partial_pdfs`. It is kept to cross-check the cell
  list in the test suite.
- **`Scattering` requires a cell periodic along all three axes** and raises `ValueError`
  otherwise. The minimum image convention is applied unconditionally, so a free surface was
  silently folded in rather than left alone.
- **`vitrum.geometry.get_dist_numba`**, the orthorhombic-only distance kernel.
  `get_dist_numba_triclinic` covers it at the same cost — a perpendicular cell needs no image
  search — and `distance_matrix` now takes that one path.
- **A cutoff can no longer be given as a bare list** such as `cutoff=[1.6, 2.4]`; pass a number
  or a dict keyed by bond or by neighbour species. It follows that the two arms of a
  same-species angle always share one cutoff.
- `vitrum.io_helpers.parse_composition` (use `pymatgen.core.Composition` directly) and
  `vitrum.mlip_functions.min_max_val`.
- The `vitrum.structure_gen` helpers `my_round`, `balance_charge`, `is_multiple`,
  `choose_count` and `random_partition` are private, `_`-prefixed. They are internals of the
  composition sampling, reachable through `GlassGenerator`.
- `Diffusion.get_van_hove_dist_correlation` and `Diffusion.get_velocity_autocorrelation`, which
  were empty stubs.
- `vitrum.persistent_homology.LocalPD`, `get_persistence_diagram` and `get_local_persistence`,
  superseded by `PersistenceDiagram`. `LocalPD` had been broken since the persistence code
  moved out of `glass_Atoms`.
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
- 1.0.1's `Scattering(use_neighborhood=True)` returned incorrect partial and total RDFs: one
  ordering of every cross pair came back as exactly zero. The backend it selected has been
  replaced by the cell list, which does not have the bug.
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