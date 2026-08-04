# Utility functions:

This is an overview of various miscellaneous utility functions, useful for different purposes when doing calculations with vitrum. These used to live together in a single `vitrum.utility` module, which has since been split into focused modules.

## Geometry

Low-level distance/PDF helpers used internally by `GlassAtoms` — useful directly when you have raw distances rather than a full `Atoms` object.

```python
from vitrum.geometry import pdf, find_min_after_peak, radial_bins

xval, gr = pdf(dist_list, volume, rrange=10, nbin=100, n_pairs=n_i * n_j)
cutoff = xval[find_min_after_peak(gr)]  # first minimum after the first peak

xval, shell_volumes = radial_bins(rrange=10, nbin=100)
```

`pdf` is the single normalisation used by `GlassAtoms.get_pdf` and both `Scattering`
backends. Omitting `n_pairs` falls back to `dist_list.size`, which over-counts like
pairs by `n / (n - 1)`; pass it explicitly.

`require_orthorhombic` is the shared guard behind the orthorhombic-cell restriction
described in [Known issues](known_issues.md). It returns the cell diagonal, or raises
`NotImplementedError` if the cell is triclinic.

::: vitrum.geometry

## Trajectory

See [Quick start](quickstart.md) for a worked `unwrap_trajectory` example.

::: vitrum.trajectory_tools

## Packing

See [Quick start](quickstart.md) for a worked `get_random_packed` example.

::: vitrum.packing

## Volume estimation

Used internally by `packing.get_random_packed` to estimate the cell volume for a target composition; can also be called directly. This module also holds the shared per-atom radii helpers (`get_packing_radii`, `guess_oxi_states`), which `packing` uses for hard-sphere overlap resolution.

```python
from vitrum.volume_estimation import get_volume

# The default estimator uses ionic radii and needs no API key or optional dependency.
volume = get_volume("SiO2", {"Si": 1, "O": 2})

# Materials Project lookups need the optional extra: pip install vitrum[volume_estimation]
volume = get_volume("SiO2", {"Si": 1, "O": 2}, vol_per_atom_source="mp")
```

The default `"ionic_radius"` estimator is calibrated on oxides. For metallic or covalent systems, or when you know the density, pass `density=` or a float volume per atom instead.
See `IONIC_PACKING_FRACTION` in `vitrum.volume_estimation` for the calibration.

::: vitrum.volume_estimation

## I/O helpers

See [Quick start](quickstart.md) for worked `correct_atom_types`/`get_LAMMPS_dump_timesteps` examples.

```python
from vitrum.io_helpers import mass_density_to_number_density, number_density_to_mass_density

number_density = mass_density_to_number_density("SiO2", density=2.2)  # atoms/Angstrom^3
density = number_density_to_mass_density("SiO2", number_density)      # g/cm^3, round-trips
```

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
