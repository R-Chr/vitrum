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

Keys a call does not need are ignored, so a single dictionary can be shared across every method; a bond with *no* entry is an error. Where both forms could apply, the explicit pair key wins over the bare-species shorthand.

```python
cutoffs = {"O": 1.6, "Na": 3.0}

coord.get_coordination_numbers("Si", ["O", "Na"], cutoffs)
coord.get_angles("Si", ["O", "Na"], cutoffs)
coord.get_neighbors("Si", cutoffs)
coord.get_bonds("Si", "O", cutoffs)
```


An `"Auto"` cutoff is resolved **once, from one frame**, and then applied to every frame, so it cannot drift along a trajectory and every method agrees on the same value. That frame is the first by default.

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

`get_angle_distribution` reports P(theta) by default. Pass `sin_normalised=True` for the P(theta)/sin(theta) convention, which divides out the solid-angle weighting that favours 90 degrees even in an uncorrelated structure. Both are in common use, so a figure should say which one it shows.

### Mixed-former Q^n analysis

`get_bridging_analysis` measures two kinds of bond — centre-bridge, which sets `n`, and former-bridge, which decides whether a bridge atom bridges. In a mixed-former glass these have different lengths, so name them separately:

```python
coord.get_bridging_analysis(
    "Si", "O", former_types=["Si", "B"],
    cutoff={("Si", "O"): 1.9, ("B", "O"): 1.6},
)
```

Every bond here shares the bridge species as its neighbour, so telling the formers apart needs the pair-keyed form; `{"O": 1.9}` names the bridge and therefore applies one cutoff to all of them. With `"Auto"`, each former's bond is resolved from its own partial PDF.

### Speciation

`get_bridging_analysis` counts bridging neighbours per **former**; `get_bridging_speciation` counts formers per **bridge**, which is the oxygen speciation. One distribution holds every class, so nothing is labelled in advance: free oxygen at `n=0`, non-bridging at `n=1`, bridging at `n=2`, tri-clusters at `n=3` and above.

```python
from vitrum.coordination import (network_connectivity)

cutoff = {("Si", "O"): 2.0, ("Na", "O"): 3.0}

qn = coord.get_bridging_analysis("Si", "O", cutoff=cutoff)
network_connectivity(qn)                                    # NC = sum(n * f_n)

o_speciation = coord.get_bridging_speciation("O", "Si", cutoff=cutoff)
o_speciation.get(1, 0.0)                                    # the NBO fraction
```

`network_connectivity` is the mean of any of these distributions, so it also turns a coordination-number distribution into an average coordination number.

## The primitive underneath: `Bonds`

Every method above is a reduction of one object — which atoms of a centre selection lie within a cutoff of which atoms of a neighbour selection. `get_bonds` exposes it, so quantities the package does not ship can be derived without reaching into internals.

```python
from vitrum import Bonds  # noqa: F401  - returned by get_bonds

bonds = coord.get_bonds("Si", "O", cutoff=1.9)[0]   # one Bonds per frame

bonds.counts()        # bonds per Si  -> get_coordination_numbers
bonds.degrees()       # bonds per O   -> the bridging test in get_bridging_analysis
bonds.lists()         # global O indices per Si -> get_neighbors
bonds.lengths(frame)  # length of every bond, for polyhedral distortion metrics
bonds.select_neighs(bonds.degrees() >= 2)           # keep only bridging oxygens
```

`centers` and `neighs` hold global atom indices, so `atoms[bonds.neighs[0]]` is a real atom. Either selection may name several species: `coord.get_bonds(["Si", "B"], "O", cutoff=1.8)` is how `get_bridging_analysis` treats a mixed-former glass. One `Bonds` holds one cutoff, so a multi-species selection has to resolve to a single value — a dict that gives Si-O and B-O different cutoffs, or an `"Auto"` that would, is rejected rather than measured at whichever came first. Call `get_bonds` once per bond there.

Bonds are stored as an edge list rather than an N x N matrix, and found with `ase.neighborlist.neighbor_list`, so cost scales with the number of bonds rather than with the square of the system size, and any cell shape works — including triclinic ones, unlike most of the rest of the package (see [Known issues](known_issues.md)).

::: vitrum.bonds

::: vitrum.coordination