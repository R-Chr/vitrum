# Migrating from GlassAtoms

`GlassAtoms` was an `ase.Atoms` subclass that carried the single-structure analysis methods.
It is **deprecated since 1.1.0 and will be removed in 2.0.0**.

Every analysis class now takes and holds plain `ase.Atoms`, so there is no wrapper type to
construct and nothing to remember to convert. A structure read with `ase.io.read` goes
straight into `Coordination`, `Scattering`, `RingAnalysis` and the rest.

Existing code keeps working: each `GlassAtoms` method still returns exactly what it always
returned, and emits a `DeprecationWarning` naming its replacement.

## Old to new

| Removed in 2.0.0 | Use instead |
| --- | --- |
| `GlassAtoms(atoms).get_dist()` | `vitrum.geometry.distance_matrix(atoms)` |
| `GlassAtoms(atoms).get_pdf(pair)` | `Scattering(atoms).get_partial_pdf(pair)` |
| `GlassAtoms(atoms).get_all_angles(...)` | `Coordination([atoms]).get_angles(...)` |
| `GlassAtoms(atoms).get_coordination_number(...)` | `Coordination([atoms]).get_coordination_numbers(..., per_atom=True)` |
| `GlassAtoms(atoms).get_bridging_analysis(...)` | `Coordination([atoms]).get_bridging_analysis(...)` |
| `GlassAtoms(atoms).get_neighbors(...)` | `Coordination([atoms]).get_neighbors(...)` |
| `GlassAtoms(atoms).get_density()` | `vitrum.io_helpers.get_density(atoms)` |
| `GlassAtoms(atoms).set_new_chemical_symbols(map)` | `vitrum.io_helpers.correct_atom_types(atoms, map)` |

The `Coordination` replacements are trajectory-aware. By default they aggregate over every
frame; pass `per_atom=True` for the raw per-centre-atom values, which come back as one entry
per frame. Wrap a single structure in a one-element list, and take element `0` to get that
frame back.

Two replacements do not return quite what the old method did:

- `Coordination.get_neighbors` returns **global** atom indices. `GlassAtoms.get_neighbors`
  returned indices into each species' own atoms, and still does.
- `Coordination.get_angles` returns one flat array of angles per frame.
  `GlassAtoms.get_all_angles` returned one array per centre atom, which is now
  `get_angles(..., per_atom=True)[0]`.

## Example

```python
# Before
from vitrum.glass_atoms import GlassAtoms

atoms = GlassAtoms(atoms)
r, gr = atoms.get_pdf(["Si", "O"])
coordination_numbers = atoms.get_coordination_number("Si", "O")
density = atoms.get_density()
```

```python
# After
from vitrum import Coordination, Scattering
from vitrum.io_helpers import get_density

scattering = Scattering(atoms)
r, gr = scattering.xval, scattering.get_partial_pdf(("Si", "O"))

coordination = Coordination([atoms])
coordination_numbers = coordination.get_coordination_numbers("Si", "O", per_atom=True)[0]

density = get_density(atoms)
```

::: vitrum.glass_atoms
