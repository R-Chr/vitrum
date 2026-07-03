# Extended ASE atoms
We have added commonly used structure analysis for individual structures of disordered materials to the ASE Atoms object. The new methods are listed below. The ASE `Atoms` are automatically converted to a `GlassAtoms` object when they are passed to the vitrum classes.

## Example usage:

```python
from vitrum.glass_atoms import GlassAtoms

atoms = GlassAtoms(atoms)  # wrap an existing ase.Atoms

# pair distribution function for a Si-O pair
r, gr = atoms.get_pdf(["Si", "O"])

# coordination number of O around each Si, with an automatically determined cutoff
coordination_numbers = atoms.get_coordination_number("Si", "O")

# density in g/cm^3
density = atoms.get_density()
```

::: vitrum.glass_atoms