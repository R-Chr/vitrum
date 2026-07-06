# Void analysis

The voids module contains functions for calculating free volume and discrete cavities ("voids") in an `Atoms` object, using a grid/probe-accessible-volume approach: a fine 3D grid is overlaid on the simulation cell, each grid point is classified as occupied or free based on its distance to the nearest atom's exclusion radius, and contiguous free grid points are clustered into discrete cavities with a periodic-boundary-aware merge step.

`VoidAnalysis` currently requires an **orthorhombic** simulation cell — a tilted/triclinic cell will raise a `ValueError`.

Included functionality:
- Free volume fraction of the cell
- Discrete cavity identification, with per-cavity volume, effective radius, and periodic-safe center
- Cavity size distribution and plotting
- Export of cavity centers as dummy pseudo-atoms for visualization alongside the structure
- Interactive 3D visualization of the void space (as a marching-cubes isosurface) together with the atoms, via `plot_3d` (requires the optional `plotly`/`scikit-image` dependencies, see [Installation](install.md))

## Example usage:

```python
from vitrum.voids import VoidAnalysis

va = VoidAnalysis(atoms, radii_scaling=1.0, probe_radius=0.0)
cavities = va.calculate(grid_spacing=0.2)

print(va.get_free_volume_fraction())
va.plot_cavity_size_distribution()
va.write_cavities("cavities.extxyz")

fig = va.plot_3d()
fig.show()
```

Like `RingAnalysis`, void analysis operates on a single structure, not a trajectory. To compare free volume across a trajectory, run `VoidAnalysis` per frame in a loop.

::: vitrum.voids
