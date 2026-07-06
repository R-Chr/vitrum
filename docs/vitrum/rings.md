# Rings

The rings class contains functions for calculating ring statistics on an Atoms objects.
The implementation of rings is an adapted version of the code for calculating rings from https://github.com/MotokiShiga/sova-cui making the code compatible with the vitrum package.

Currently supported ring types are:
- Guttman
- King
- Primitive

Included functions for ring objects:
- Center
- Size
- Perimeter
- Roundness
- Roughness
- Radius of gyration

## Example usage:

```python
from vitrum.rings import RingAnalysis

ring_funcs = RingAnalysis(atoms, included_atoms=["Si", "O"], bonding_dict=[("Si", "O")])
rings = ring_funcs.calculate(criterion="guttman")  # or "king" / "primitive"
sizes = ring_funcs.get_ring_size_distribution()
```

::: vitrum.rings