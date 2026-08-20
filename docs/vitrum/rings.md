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

## Example usage

```python
from vitrum.rings import RingAnalysis

ring_funcs = RingAnalysis(atoms, included_atoms=["Si", "O"], bonding_dict=[("Si", "O")])
rings = ring_funcs.calculate(criterion="guttman")  # or "king" / "primitive"
sizes = ring_funcs.get_ring_size_distribution()
```

## Deciding a bond: covalent radii, or an explicit cutoff

By default the bond graph comes from covalent radii scaled by `radii_factor`, as above. Pass
`cutoff` to build it from a distance instead, in the same grammar `Coordination` uses:
`"Auto"`, a number, or a dict keyed by species pair (`{("Si", "O"): 1.9}`) or by a single
species as shorthand. It is resolved once against `bonding_dict`, or against every species
pair in the structure if `bonding_dict` is None:

```python
rings = ring_funcs.calculate(criterion="guttman", cutoff={("Si", "O"): 1.9})
rings = ring_funcs.calculate(criterion="guttman", cutoff="Auto")
```

This route works for any cell, including a triclinic one, unlike the `radii_factor` default.

::: vitrum.rings