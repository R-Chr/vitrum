# Known issues

This page tracks confirmed correctness bugs and unimplemented code paths that
have not yet been fixed.

## The `PersistenceDiagram` filtration is not periodic

`PersistenceDiagram.calculate` passes the atom positions to the weighted alpha-shape
construction as a finite point cloud. It uses no periodic images and no cell
information, so every atom near a cell face contributes loops and voids that exist
only because the structure was truncated there.

This is a surface effect and scales with the surface-to-volume ratio of the cell: it
is worst for the few-hundred-atom cells typical of ab-initio glass models. Treat the
diagrams as comparable to each other at fixed cell size and shape, not as absolute
loop/void statistics of the bulk.

## Orthorhombic cells only

Trajectory unwrapping (`unwrap_trajectory`), void grids (`build_void_grid`,
`VoidAnalysis`) and the homogeneity checker still apply the minimum image convention using
only the cell diagonal, so they are valid only for orthorhombic cells.
This is checked rather than assumed: passing a triclinic cell raises `NotImplementedError`
via `vitrum.geometry.require_orthorhombic` instead of silently returning
plausible-looking but wrong numbers. Full triclinic support is not implemented for these
routines. Positions need not be wrapped into the cell.

The following are **not** subject to this restriction:

- **Distances.** `vitrum.geometry.distance_matrix` takes any cell as of 1.1.0.
- **`Scattering`.** Its cell-list PDF backend works on a general cell. `rrange` is bounded by
  the perpendicular cell widths rather than the cell lengths, which is the correct
  minimum-image limit. `Scattering` does require a cell periodic along all three axes as of
  1.1.0.
- **Ring metrics.** `Ring` unwraps bond by bond with `ase.geometry.find_mic` and wraps its
  centre through the cell's own fractional coordinates, both of which take a general cell.
- **Bond graphs.** `Coordination`, the underlying `vitrum.bonds.Bonds`/`_Frame`, and `find_rings`/
  `RingAnalysis.calculate` given an explicit `cutoff` build their bond graph entirely
  with `ase.neighborlist.neighbor_list`, which takes a general cell matrix, so
  `cutoff="Auto"` and every other cutoff spelling work on a triclinic cell. `find_rings`'s
  covalent-radii default (`cutoff=None`) was never subject to this restriction in the first
  place. It already built its graph from a general cell matrix.

## `Diffusion` limitations

Not bugs, but the estimator is less careful than the rest of the package and the
module has no test coverage beyond its input guards.

- `get_mean_square_displacements` measures displacements from a **single time
  origin** (the first frame) rather than averaging over multiple origins. This is
  noisier than the standard windowed estimator, especially at long lag times where
  the single-origin estimate rests on one sample per atom.
- `get_van_hove_self_correlation` uses its `t_window` as both the stride and the
  window, so time origins never overlap.


## `batch_active` is stale and largely untested

The `vitrum.batch_active` module drives VASP/LAMMPS active-learning workflows
through FireWorks. It requires external services that are not covered by the
test suite, and it is broken in several places. Treat it as unsupported.

- `balace.run_train_pace()` raises `TypeError`. It calls `train_pace(self)`,
  which then does `**pace_kwargs` with `pace_kwargs=None`.
- `train_pace` submits its workflow twice: once itself
  (`workflow.py`, `self.lp.add_wf(wf)`) and again in the caller
  (`learning.py`, `run_train_pace`).
- `train_grace` returns `None`, but `run_train_grace` assigns the result to
  `directory` and appends it to `self.runs["potential"]`. Grace training is not
  automated. The function only creates a directory and prints instructions to
  place a trained model there by hand.
- `train_pace` and `train_grace` are module-level functions that still take
  `self` as their first parameter; they were lifted out of a class and never
  adapted.
- `load_config` applies arbitrary YAML keys to the instance with `setattr`, so you
  cannot tell which attributes a `balace` object has without running it.

