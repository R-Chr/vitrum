# Coordination
Class for coordination analysis, i.e. bond angle distributions, coordination number, bridging analysis etc. `Coordination` takes a **list** of plain ASE `Atoms` objects (e.g. multiple frames of a trajectory) and aggregates statistics over them.

## Specifying a cutoff

A cutoff belongs to a **bond**, not to an atom, and every method spells one the same way:

| spelling | meaning |
| --- | --- |
| `"Auto"` | resolve each bond from the first minimum after the first peak of that bond's partial PDF |
| `1.9` | one cutoff for every bond the call measures |
| `{("Si", "O"): 1.6}` | one cutoff per bond, keyed by the pair; key order does not matter |
| `{"O": 1.6}` | shorthand for the bond from the centre to that species |
| `[1.6, 3.0]` | one per neighbour type, in the order they were named |

Keys a call does not need are ignored, so a single dictionary can be shared across every method; a bond with *no* entry is an error. Where both forms could apply, the explicit pair key wins over the bare-species shorthand.

```python
cutoffs = {"O": 1.6, "Na": 3.0}

coord.get_coordination_numbers("Si", ["O", "Na"], cutoffs)
coord.get_angles("Si", ["O", "Na"], cutoffs)
coord.get_neighbors("Si", cutoffs)
coord.get_bonds("Si", "O", cutoffs)
```

The list form is positional against the neighbour-type argument, so only `get_angles`, `get_angle_distribution` and `get_coordination_numbers` accept one — `get_neighbors` has no such argument to order it against and rejects a list rather than guessing.

An `"Auto"` cutoff is resolved **once, from the first frame**, and then applied to every frame, so it cannot drift along a trajectory and every method agrees on the same value.

## Example usage:

```python
from vitrum.coordination import Coordination
coord = Coordination(atoms)  # atoms: a list of frames, or a single Atoms object

# distributions aggregated over the whole trajectory
angles, dist = coord.get_angle_distribution("Si", "O", cutoff=2)
cn = coord.get_coordination_numbers("Si", "O")            # {n: fraction}
qn = coord.get_bridging_analysis("Si", "O")               # Q^n speciation

# the same quantities raw, one entry per frame
per_atom = coord.get_coordination_numbers("Si", "O", per_atom=True)
per_atom_qn = coord.get_bridging_analysis("Si", "O", per_atom=True)
raw_angles = coord.get_angles("Si", ["O", "O"])
neighbors = coord.get_neighbors("Si", cutoff={"O": 2.0, "Na": 3.0})
```

`per_atom=True` means the same thing everywhere: one entry per frame, holding one value per centre atom instead of the trajectory-wide summary. `get_angles` returns one flat array of angles per frame by default; `get_angles(..., per_atom=True)` groups those angles by the centre atom they were measured at.

`get_neighbors` returns **global** atom indices, so an entry indexes straight into the frame. (The deprecated `GlassAtoms.get_neighbors` returns indices into each species' own atoms instead.)

### Mixed-former Q^n analysis

`get_bridging_analysis` measures two kinds of bond — centre-bridge, which sets `n`, and former-bridge, which decides whether a bridge atom bridges. In a mixed-former glass these have different lengths, so name them separately:

```python
coord.get_bridging_analysis(
    "Si", "O", former_types=["Si", "B"],
    cutoff={("Si", "O"): 1.9, ("B", "O"): 1.6},
)
```

Every bond here shares the bridge species as its neighbour, so telling the formers apart needs the pair-keyed form; `{"O": 1.9}` names the bridge and therefore applies one cutoff to all of them. With `"Auto"`, each former's bond is resolved from its own partial PDF.

## The primitive underneath: `Bonds`

Every method above is a reduction of one object — which atoms of a centre selection lie within a cutoff of which atoms of a neighbour selection. `get_bonds` exposes it, so quantities the package does not ship can be derived without reaching into internals.

```python
from vitrum import Bonds  # noqa: F401  - returned by get_bonds

bonds = coord.get_bonds("Si", "O", cutoff=1.9)[0]   # one Bonds per frame

bonds.counts()        # bonds per Si  -> get_coordination_numbers
bonds.degrees()       # bonds per O   -> the bridging test in get_bridging_analysis
bonds.lists()         # global O indices per Si -> get_neighbors
bonds.select_neighs(bonds.degrees() >= 2)           # keep only bridging oxygens
bonds.matrix()        # dense boolean adjacency, materialised on request
```

`centers` and `neighs` hold global atom indices, so `atoms[bonds.neighs[0]]` is a real atom. Either selection may name several species: `coord.get_bonds(["Si", "B"], "O", cutoff=1.8)` is how `get_bridging_analysis` treats a mixed-former glass.

Bonds are stored as an edge list rather than an N x N matrix, and found with `ase.neighborlist.neighbor_list`, so cost scales with the number of bonds rather than with the square of the system size, and any cell shape works — including triclinic ones, unlike most of the rest of the package (see [Known issues](known_issues.md)).

::: vitrum.bonds

::: vitrum.coordination