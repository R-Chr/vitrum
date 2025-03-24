Several examples on the usage of the vitrum package can be found the the `examples` folder on the vitrum github

## Atomic structures
### Reading in atomic structures
Using many of the vitrum classes requires having the atomic structure of a material. For this we use the ASE atoms object. The atomic structures outputed from a simulation, here conducted in LAMMPS can be read from a dump file using the following line:

```
from ase.io import read
atoms = read("md.lammpstrj", index=":" , format="lammps-dump-text")
```

Often the chemical symbols of the atoms in the atoms object are not the same as the chemical symbols used in the simulation. This can be corrected using the `correct_atom_types` function. For example, if the chemical symbols used in the simulation are ['Na', 'O', 'Si'], the following line can be used to correct the symbols:

```
from vitrum.utility import correct_atom_types
corr_atoms_dic = {1: 'Na', 2: 'O', 3:'Si'}
correct_atom_types(atoms, corr_atoms_dic)
```

### Generating random structures
The `get_random_packed` function can be used to generate random structures. For example, to generate a random structure with 1000 atoms, the following line can be used:

```
from vitrum.utility import get_random_packed
atoms = get_random_packed(composition='SiO2', density=2.2, target_atoms=1000)
```

The `composition` parameter can be used to specify the chemical composition of the structure. The `get_random_packed` function includes several parameters to tailor random structure generation according to your given needs.


## Scattering functions
The `scattering` class contains functions for calculating scattering functions of materials, as averaged over a list of Extended ASE Atoms objects.

```
from vitrum.scattering import scattering
scattering_funcs = scattering(atoms)
```

To calculate the total neutron radial distribution function of a material, the following line can be used:

```
G_r = scattering_funcs.get_total_rdf(type="neutron")
```


## Diffusion analysis
The `diffusion` class contains functions for calculating diffusion properties of materials, as averaged over a list of ASE Atoms objects.

```
from vitrum.scattering import diffusion
diffusion_funcs = diffusion(atoms, sample_times = timesteps)
```

To calculate the mean squared displacement of a material, the following line can be used:

```
msd = diffusion_funcs.get_mean_square_displacements()
```

To get the timesteps at which the mean squared displacement was calculated, one of the utility functions can be used:

```
from vitrum.utility import get_LAMMPS_dump_timesteps
timesteps = get_LAMMPS_dump_timesteps('md.lammpstrj')
```

The diffusion class assumes that trajectories are unwrapped, and not wrapped according to PBC. To unwrap a trajectory one of the utility functions can be used:

```
from vitrum.utility import unwrap_trajectory
atoms = unwrap_trajectory(atoms)
```