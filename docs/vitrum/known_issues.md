# Known Issues

This page tracks confirmed correctness bugs and unimplemented code paths that
have not yet been fixed.

## `persistent_homology.py` — `LocalPD` and `get_local_persistence`

Quarantined: both raise `NotImplementedError` 

## `PersistenceDiagram` — the filtration is not periodic

`PersistenceDiagram.calculate` passes the atom positions to the weighted alpha-shape
construction as a finite point cloud. It uses no periodic images and no cell
information, so every atom near a cell face contributes loops and voids that exist
only because the structure was truncated there.

This is a surface effect and scales with the surface-to-volume ratio of the cell: it
is worst for the few-hundred-atom cells typical of ab-initio glass models. Treat the
diagrams as comparable to each other at fixed cell size and shape, not as absolute
loop/void statistics of the bulk.

## `Scattering.get_total_rdf(type="xray")`

Not implemented: raises `NotImplementedError`. Previously this printed a message
and returned an array of zeros.

Use `type="approx_xray"` for the Q-independent atomic-number approximation, or
`type="neutron"`. Note that `get_structure_factor(type="xray")` *is* implemented
— the limitation only affects the real-space RDF.

## Orthorhombic cells only

Distance calculations, trajectory unwrapping, void grids and ring centres all
apply the minimum image convention using only the cell diagonal, so they are
valid only for orthorhombic cells.

As of 1.1.0 this is checked rather than assumed: passing a triclinic cell raises
`NotImplementedError` via `vitrum.geometry.require_orthorhombic` instead of
silently returning plausible-looking but wrong numbers. Full triclinic support
is not implemented.

## `Diffusion` — limitations

Not bugs, but the estimator is less careful than the rest of the package and the
module has no test coverage beyond its input guards.

- `get_mean_square_displacements` measures displacements from a **single time
  origin** (the first frame) rather than averaging over multiple origins. This is
  noisier than the standard windowed estimator, especially at long lag times where
  the single-origin estimate rests on one sample per atom.
- `get_van_hove_self_correlation` normalises by the total atom count rather than by
  the number of atoms of the target species, so the histogram it returns is scaled
  down by the target species' concentration. Its `t_window` is used as both the
  stride and the window, so time origins never overlap.
- `get_van_hove_dist_correlation` and `get_velocity_autocorrelation` are stubs that
  return `None`.


## `batch_active` — stale and largely untested

The `vitrum.batch_active` module drives VASP/LAMMPS active-learning workflows
through FireWorks. It requires external services that are not covered by the
test suite, and it is known to be broken in several places:

- `balace.run_train_pace()` raises `TypeError`. It calls `train_pace(self)`,
  which then does `**pace_kwargs` with `pace_kwargs=None`.
- `train_pace` submits its workflow twice: once itself
  (`workflow.py`, `self.lp.add_wf(wf)`) and again in the caller
  (`learning.py`, `run_train_pace`).
- `train_grace` returns `None`, but `run_train_grace` assigns the result to
  `directory` and appends it to `self.runs["potential"]`. Grace training is not
  automated — the function only creates a directory and prints instructions to
  place a trained model there by hand.
- `train_pace` and `train_grace` are module-level functions that still take
  `self` as their first parameter; they were lifted out of a class and never
  adapted.
- `load_config` applies arbitrary YAML keys to the instance with `setattr`, so
  the attribute surface of `balace` cannot be determined statically.

Treat this module as unsupported. The analysis modules (`scattering`,
`coordination`, `rings`, `voids`, `glass_atoms`) are the tested, supported part
of the package.
