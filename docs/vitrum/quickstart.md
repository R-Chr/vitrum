More worked examples live in the `examples` folder of the [vitrum repository](https://github.com/R-Chr/vitrum/tree/main/examples).

## Atomic structures
### Reading in atomic structures
Most vitrum classes take an ASE `Atoms` object, or a list of them for a trajectory. A trajectory written by LAMMPS reads in from its dump file:

```
from ase.io import read
atoms = read("md.lammpstrj", index=":" , format="lammps-dump-text")
```

A dump file carries numeric atom types rather than chemical symbols, so the symbols ASE guesses are usually wrong. `correct_atom_types` maps them back. For types 1, 2 and 3 standing for Na, O and Si:

```
from vitrum.io_helpers import correct_atom_types
corr_atoms_dic = {1: 'Na', 2: 'O', 3:'Si'}
correct_atom_types(atoms, corr_atoms_dic)
```

### Generating random structures
`get_random_packed` builds a random structure of a given composition and density. For 1000 atoms of SiO2:

```
from vitrum.packing import get_random_packed
atoms = get_random_packed(composition='SiO2', density=2.2, target_atoms=1000)
```

It takes several more parameters to shape the packing, see [Utilities](utilities.md).

Atoms are placed by resolving hard-sphere overlaps, passing `charge_ordering=1.0` keeps like-charged ions apart, so anions end up between cations:

```
atoms = get_random_packed(composition='SiO2', density=2.2, target_atoms=1000, charge_ordering=1.0)
```

It is off by default, and has no effect on compositions with no charge-balanced oxidation states, such as metals and alloys.


## Scattering functions
The `Scattering` class contains functions for calculating scattering functions of materials, as averaged over a list of Extended ASE Atoms objects.

```
from vitrum.scattering import Scattering
scattering_funcs = Scattering(atoms)
```

The total neutron radial distribution function:

```
G_r = scattering_funcs.get_total_rdf(type="neutron")
```


## Coordination analysis
The `Coordination` class contains functions for calculating bond angle distributions and coordination numbers, as averaged over a list of Extended ASE Atoms objects.

```
from vitrum.coordination import Coordination
coord_funcs = Coordination(atoms)
```

The O-Si-O bond angle distribution, Si being the centre atom:

```
angles, dist = coord_funcs.get_angle_distribution("Si", "O", cutoff=2)
```

The coordination number distribution of O around Si:

```
coordination_numbers = coord_funcs.get_coordination_numbers("Si", "O")
```


## Ring analysis
The `RingAnalysis` class finds and analyzes rings (Guttman-type) in a single structure.

```
from vitrum.rings import RingAnalysis
ring_funcs = RingAnalysis(atoms[0], included_atoms=["Si", "O"], bonding_dict=[("Si", "O")])
rings = ring_funcs.calculate()
```

The distribution of ring sizes:

```
sizes = ring_funcs.get_ring_size_distribution()
```


## Diffusion analysis
The `Diffusion` class contains functions for calculating diffusion properties of materials, as averaged over a list of ASE Atoms objects.

```
from vitrum.diffusion import Diffusion
diffusion_funcs = Diffusion(atoms, sample_times = timesteps)
```

The mean squared displacement:

```
msd = diffusion_funcs.get_mean_square_displacements()
```

The timesteps those displacements belong to come from a utility function:

```
from vitrum.io_helpers import get_LAMMPS_dump_timesteps
timesteps = get_LAMMPS_dump_timesteps('md.lammpstrj')
```

By default, `Diffusion` assumes the trajectory is PBC-wrapped and unwraps it internally before computing displacements. If you've already unwrapped it yourself (e.g. via `vitrum.trajectory_tools.unwrap_trajectory`), pass `wrapped=False` to avoid unwrapping twice:

```
diffusion_funcs = Diffusion(atoms, sample_times=timesteps, wrapped=False)
```