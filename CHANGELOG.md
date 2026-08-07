# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-08-05

This release corrects several bugs that produced silently wrong numbers.
**Ring statistics and topology metrics, running coordination numbers, partial PDFs for like
pairs, broadened RDFs, anything computed with `Scattering(use_neighborhood=True)`, on an NPT
trajectory, or on a structure whose positions were not wrapped into the cell, should be
recomputed.**

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
- `Coordination.get_bridging_analysis`, `get_angles` and `get_neighbors`, the
  trajectory-aware replacements for the corresponding `GlassAtoms` methods (see Deprecated
  below). `get_bridging_analysis` returns the Q^n fractions aggregated over the trajectory.
- A `per_atom=True` flag on `get_coordination_numbers`, `get_bridging_analysis` and
  `get_angles`. It means the same thing on all three: one entry per frame, holding the raw
  per-centre-atom values instead of the trajectory-wide summary.
- `Coordination.get_neighbors` accepts `cutoff="Auto"`, which every other method already
  did, and resolves one cutoff per neighbour species.
- **A cutoff can be given per bond, as `cutoff={("Si", "O"): 1.6, ("B", "O"): 1.4}`.** A
  cutoff is a property of a bond rather than of an atom, so this is now the primary
  spelling; a bare species key such as `{"O": 1.6}` remains as shorthand for the bond from
  the centre to that species, and the explicit pair wins where both could apply. Key order
  is irrelevant — `("O", "Si")` and `("Si", "O")` are the same bond.
- **`get_bridging_analysis` judges each network former at its own cutoff.** Previously one
  cutoff, taken from the centre-bridge pair, decided every bond including the
  former-bridge ones, so a borosilicate measured B-O against the Si-O bond length. It can
  now be given `cutoff={("Si", "O"): 1.9, ("B", "O"): 1.6}`, and `"Auto"` resolves each
  former's bond from its own partial PDF. A single number behaves exactly as before.
- `Coordination` accepts a single `Atoms` object, or any iterable of frames, in place of a
  list. 
- **`vitrum.bonds.Bonds` and `Coordination.get_bonds`.** Coordination numbers, neighbour
  lists, bond angles and Q^n speciation are all reductions of one object: which atoms of a
  centre selection lie within a cutoff of which atoms of a neighbour selection. That object
  is now named, and `get_bonds` returns it, so quantities the package does not ship — which
  polyhedra share an edge rather than a corner, custom speciation rules — can be derived
  without reaching into internals. `counts()` gives coordination numbers, `degrees()` the
  mirror count per neighbour, `lists()` the neighbours of each centre as global atom
  indices, and `select_neighs()` filters on a per-neighbour test. Either selection may name
  several species.
- `vitrum.geometry.distance_matrix` and `vitrum.geometry.partial_pdf`. `partial_pdf` is now
  the single definition of a partial g(r) for the dense backend, shared by `Scattering` and
  by automatic cutoff resolution in `Coordination`, which previously held separate copies of
  the pair-counting convention.
- `vitrum.io_helpers.get_density`, for the mass density of a built structure.
- **`find_rings` and `RingAnalysis.calculate` accept `cutoff`**, building the bond graph from
  a distance instead of covalent radii, in the same grammar `Coordination` uses — `"Auto"`, a
  number, or a dict keyed by species pair or by a single species as shorthand. Resolved once
  against `bonds`/`bonding_dict`, or every species pair present if that is None. `cutoff=None`
  (the default) keeps the previous covalent-radii `NeighborList` search unchanged.

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
- **`GlassAtoms.get_all_angles` now requires exactly two neighbour types.** An angle is
  spanned by two neighbours, but a longer `neigh_types` list was accepted and everything
  past the second entry silently ignored, so `["O", "Na", "F"]` returned only O–centre–Na
  angles. This also applies to `Coordination.get_angle_distribution`.
- **`cutoff="Auto"` is now resolved once from the first frame in every `Coordination`
  method.** `get_coordination_numbers` already did this; `get_angle_distribution`,
- **`Coordination.get_neighbors` returns global atom indices**, so an entry indexes straight
  into the frame. It previously returned indices into each species' own atoms, which callers
  had to map back by hand — as `get_bridging_analysis` did internally. The deprecated
  `GlassAtoms.get_neighbors` keeps the old species-relative convention.
- **`Coordination.get_angles` returns one flat array of angles per frame**, matching the
  other raw accessors. It previously returned one array per centre atom with the frames
  merged together, so per-frame results could not be recovered. The per-centre grouping is
  now `get_angles(..., per_atom=True)`.
- Only the exact string `"Auto"` is accepted as a cutoff. `cutoff="auto"` previously
  reported a confusing list-length error from `get_angles`, and died inside NumPy comparing
  a float array against a string from `get_coordination_numbers`. A boolean cutoff is
  rejected rather than read as 1 A, and NumPy scalars such as `np.int64` are accepted.
- **Every method validates a cutoff the same way**, through one shared resolver rather than
  each method's own branch. An unusable *type* raises `TypeError` (which is what
  `get_neighbors` already did, and what the deprecated `GlassAtoms.get_neighbors`
  documents), an unusable *value* such as `"auto"` or a mismatched list length raises
  `ValueError`, and a dict with no entry for a bond raises `KeyError`. Previously a bad
  cutoff type gave `TypeError` from `get_neighbors` but `ValueError` elsewhere.
- `Coordination.get_neighbors` rejects a bare list rather than accepting one. Nothing in
  the call orders it, so it could only have been read positionally against the sorted
  species — silently wrong for a structure of different composition. Use a dict.
- `get_angle_distribution` raises when no angles were found at all, instead of returning an
  array of `NaN` from a zero-count `density=True` histogram.
- `Coordination` methods report an absent species the same way regardless of entry point.
  `get_coordination_numbers` had its own message ("not found in structure"); every method
  now uses the shared one ("not present in the structure").
- **`Coordination` no longer builds an N x N distance matrix.** Bonds are found with a
  periodic KD-tree (`scipy.spatial.cKDTree`, already a dependency) and held as an edge list,
  so both time and memory scale with the number of bonds rather than with the square of the
  system size. Measured against the previous dense implementation, per method call on the
  shipped 3000-atom trajectory: coordination numbers 3.4x faster, Q^n speciation 2.7x, bond
  angles 1.8x, neighbour lists 1.6x; at 6000 atoms, 7.6x / 5.9x / 3.5x / 3.3x. The larger
  change is the ceiling — a 24,000-atom cell needed 4.6 GB for the matrix alone and is now
  analysed in under half a second at 0.38 GB peak. `cutoff="Auto"` resolution takes the same
  path, using a short-range PDF at the unchanged 0.1 A bin width and widening only when the
  first minimum lands too near the end to be trusted. A structure with no periodic cell
  still uses a dense matrix, which is the only remaining case that needs one. **No number
  changes**: both backends were checked array-for-array against the previous implementation.
- **The KD-tree bond backend above is replaced with `ase.neighborlist.neighbor_list`.** This
  removes the orthorhombic-cell restriction from `Coordination`, `vitrum.bonds.Bonds`, and
  `cutoff="Auto"` resolution generally — including inside `find_rings`/`RingAnalysis`, see
  Added — so they now work on a triclinic cell, which `scipy.spatial.cKDTree`'s `boxsize`
  could not represent. `Bonds` gains an `offsets` attribute, the per-bond periodic shift
  vector, and a structure with no periodic cell is handled directly rather than falling back
  to a dense matrix. **No number changes** on an orthorhombic cell, checked the same way as
  the KD-tree swap above; two atoms coincident at zero separation now bond correctly, which
  the dense path's old `distance > 0` self-exclusion got wrong.
- **The opportunistic dense-matrix fast path is removed entirely from `vitrum.bonds`.** It
  only ever activated when a distance matrix happened to already be cached on the same
  `_Frame`, which no caller does; `neighbor_list` is faster at every system size measured
  regardless. `Coordination` and `find_rings`/`RingAnalysis` with an explicit `cutoff` no
  longer touch a dense matrix or `vitrum.geometry.distance_matrix` at all, so nothing in
  that path is limited to an orthorhombic cell any more. **No number changes.**
- `Coordination.get_angles` measures every angle in one batched ASE call instead of one call
  per centre atom.
- `rings.check_ring_is_periodic` is renamed `ring_closes_in_cell`, which is what it
  returns; the old name stated the opposite of its behaviour.
- `Compositions.get_structures(max_atoms=...)` and `gen_random_glasses` now check the cell
  size before packing rather than after, so rejected compositions cost nothing.
  `gen_random_glasses` takes its previously hardcoded 200-atom ceiling as `max_atoms`.
- `Scattering` raises naming the element when a species has no tabulated neutron scattering
  length, instead of failing inside NumPy with an unrelated shape error. `Coordination`
  rejects an empty `atoms_list` rather than raising `IndexError` on first use.

### Deprecated

- **`GlassAtoms` is deprecated and will be removed in 2.0.0.** Every analysis class now
  takes and holds plain `ase.Atoms`, so there is no second atoms type to convert to or
  remember. Existing code keeps working unchanged: each `GlassAtoms` method still returns
  exactly what it returned before, and warns with the name of its replacement.

  **This move changes no numbers.** Every method body was carried over as-is; the outputs
  of `Scattering`, `Coordination` and the `GlassAtoms` methods themselves were checked
  array-for-array against the previous implementation.

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

  See [Migrating from GlassAtoms](docs/vitrum/glass_atoms.md).

  Note that the deprecated methods keep the conventions they shipped with, so two of them
  no longer match their replacements exactly: `GlassAtoms.get_neighbors` still returns
  species-relative indices, and `GlassAtoms.get_all_angles` still groups its angles by
  centre atom. See Changed above.

  Known, unchanged for now, to be fixed in 2.0.0:

  - `get_bridging_analysis` applies a single centre-bridge cutoff to every former-bridge
    bond, so in a mixed-former glass the second former's bonds are judged by the first
    former's bond length. Pass an explicit `cutoff` when the two differ enough to matter.

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
- Ring topology metrics were wrong for any ring spanning more than half the cell. Positions
  were unwrapped by taking each atom's minimum image from the *first* ring atom, which
  picks the wrong image once the ring is wider than that. A planar 12-ring of radius 4.5 Å
  in a 12 Å cell — sitting entirely inside it, touching no face — measured a perimeter of
  42.8 Å against a true 27.95 Å, and a roundness of 0.937 rather than 1. Rings are now
  unwrapped bond by bond along the ring, which holds at any ring size, and every metric on
  `Ring` is exact for the case above.
- Rings larger than the primary cell folded onto their own periodic image when their
  indices were mapped back, producing "rings" that pass through the same atom several
  times: a one-atom cell searched with `repeat=(3, 3, 3)` returned `[0, 0, 0, 0]` and
  reported it as a 4-ring of zero area. These are now discarded with a warning to increase
  `repeat`.

**Coordination**

- `Coordination.get_angles` double-counted an angle when its two arms named the same
  species but were given different cutoffs (`neigh_types=["O", "O"]`,
  `cutoff=[1.6, 2.0]`): the two arm neighbour lists then differ even though the species are
  the same, so a pair inside the smaller cutoff was emitted as both `(x, y)` and `(y, x)`
  from the branch meant for two *different* species. Each qualifying pair is now counted
  once regardless of which arm it was found through. Two equal cutoffs on the same species,
  the previously-tested case, are unaffected. `get_angle_distribution`, which pools
  `get_angles`, inherited the same fix.

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
- The minimum image convention subtracted at most one cell length, so any structure whose
  positions lie further outside the cell than that — routine in unwrapped LAMMPS or extxyz
  trajectories — got silently wrong distances, and with them wrong `g(r)`, coordination
  numbers, Qₙ speciation, bond angles and structure factors. Two atoms 1.0 Å apart in a
  10 Å cell were reported 11.0 Å apart. Distances no longer depend on whether the input was
  wrapped.
- `gaussian_broadening` applied the truncation kernel `G(r−r′) − G(r+r′)` directly to
  `g(r)`. That kernel broadens the odd function `r·g(r)`, so the `r′/r` weight was missing
  and a coordination shell's `∫r²g dr` grew with the broadening: 8.3% at `Q_max = 5 Å⁻¹`,
  0.9% at 15 Å⁻¹. The shell area is now conserved.
- `Scattering` took its orthorhombic check and default `rrange` from the first frame alone,
  so a later frame of an NPT run could be binned past its own half-cell length, or be
  triclinic, without being caught. Every frame is now checked.
- `unwrap_trajectory` used the first frame's cell for the whole trajectory. Steps are now
  taken in fractional coordinates and converted with the current frame's cell, so an atom
  held at fixed fractional coordinates through an NPT cell rescaling no longer registers as
  having diffused.

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