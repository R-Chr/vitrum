# Utility functions:

This is an overview of various miscellaneous utility functions, useful for different purposes when doing calculations with vitrum. These used to live together in a single `vitrum.utility` module, which has since been split into focused modules.

## Geometry

Low-level distance/PDF helpers shared by the analysis classes — useful directly when you
want one quantity without constructing a class, or when you have raw distances rather than
a full `Atoms` object.

```python
from vitrum.geometry import distance_matrix, partial_pdf, pdf, find_min_after_peak, radial_bins

# minimum-image distance matrix for a structure (any cell, including triclinic)
distances = distance_matrix(atoms)

# one partial g_ab(r) from that matrix, without building a Scattering object
xval, gr = partial_pdf(distances, atoms.get_chemical_symbols(), atoms.get_volume(), ("Si", "O"))
cutoff = xval[find_min_after_peak(gr)]  # first minimum after the first peak

# the raw primitives, if you already have a list of distances
xval, gr = pdf(dist_list, volume, rrange=10, nbin=100, n_pairs=n_i * n_j)
xval, shell_volumes = radial_bins(rrange=10, nbin=100)
```

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
from vitrum.io_helpers import (
    get_density, mass_density_to_number_density, number_density_to_mass_density
)

number_density = mass_density_to_number_density("SiO2", density=2.2)  # atoms/Angstrom^3
density = number_density_to_mass_density("SiO2", number_density)      # g/cm^3, round-trips
get_density(atoms)  # g/cm^3; mass density of an actual structure, from its masses and cell volume
```

The two conversions work from a composition, before a structure exists; `get_density` takes
a built `Atoms` object. It replaces `GlassAtoms.get_density`.

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

## Visualization

Renders `ase.Atoms` structures to static images or Jupyter widgets via OVITO. Requires the
optional `visualization` extra, see [Installation](install.md).

```python
from vitrum.visualization import StructureRenderer

renderer = StructureRenderer(atoms, bonds={("Si", "O"): 2.0})
renderer.render(filename="structure.png")
```

::: vitrum.visualization
