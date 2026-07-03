# Utility functions:

This is an overview of various miscellaneous utility functions, useful for different purposes when doing calculations with vitrum. These used to live together in a single `vitrum.utility` module, which has since been split into focused modules.

## Geometry

Low-level distance/PDF helpers used internally by `GlassAtoms` — useful directly when you have raw distances rather than a full `Atoms` object.

```python
from vitrum.geometry import pdf, find_min_after_peak

xval, gr = pdf(dist_list, volume, rrange=10, nbin=100)
cutoff = xval[find_min_after_peak(gr)]  # first minimum after the first peak
```

::: vitrum.geometry

## Trajectory

See [Quick start](quickstart.md) for a worked `unwrap_trajectory` example.

::: vitrum.trajectory

## Packing

See [Quick start](quickstart.md) for a worked `get_random_packed` example.

::: vitrum.packing

## Volume estimation

Used internally by `packing.get_random_packed` to estimate the cell volume for a target composition; can also be called directly.

```python
from vitrum.volume_estimation import get_volume

# estimate volume purely from covalent radii, no external API required
volume = get_volume("SiO2", {"Si": 1, "O": 2}, vol_per_atom_source="covalent_radius")
```

::: vitrum.volume_estimation

## I/O helpers

See [Quick start](quickstart.md) for worked `correct_atom_types`/`get_LAMMPS_dump_timesteps` examples.

::: vitrum.io_helpers

## Structure validation

Sanity checks for generated or simulated structures.

```python
from vitrum.structure_validation import homogeneity_checker, dimer_checker

is_homogeneous = homogeneity_checker(atoms, grid_density=(3, 3, 3))
has_too_many_dimers = dimer_checker(atoms, bond_length=2.0, num_allowed=2)
```

::: vitrum.structure_validation

## Comparison

```python
from vitrum.comparison import r_chi

simulated = {"x": sc.xval, "y": rdf}
experimental = {"x": exp_r, "y": exp_gr}
rchi, common_x, y_sim, y_exp = r_chi(simulated, experimental)
```

::: vitrum.comparison
