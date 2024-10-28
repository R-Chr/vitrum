Several examples on the usage of the vitrum package can be found the the `examples` folder on the vitrum github

## Atomic structures
### Reading in atomic structures
Using many of the vitrum classes requires having the atomic structure of a material. For this we use the ASE atoms object. The atomic structures outputed from a simulation, here conducted in LAMMPS can be read from a dump file using the following line:

```
from ase.io import read
atoms = read("md.lammpstrj", index=":" , format="lammps-dump-text")
```

Several of the vitrum classes require extra utility not included in the ASE atoms object. These are implemented in the `glass_Atoms` class. To use the class, simply convert the atoms object to a glass_Atoms object using the following line:

```
from vitrum.glass_Atoms import glass_Atoms
atoms = [glass_Atoms(atom) for atom in atoms]
```

Often the chemical symbols of the atoms in the atoms object are not the same as the chemical symbols used in the simulation. This can be corrected using the `set_new_chemical_symbols` method of the glass_Atoms class. For example, if the chemical symbols used in the simulation are ['Na', 'O', 'Si'], the following line can be used to correct the symbols:

```
corr_atoms_dic = {1: 'Na', 2: 'O', 3:'Si'}
for atom in atoms:
    atom.set_new_chemical_symbols(corr_atoms_dic)
```

### Generating random structures
The `get_random_packed` function can be used to generate random structures. For example, to generate a random structure with 1000 atoms, the following line can be used:

```
from vitrum.utility import get_random_packed
atoms = get_random_packed(composition='SiO2', density=2.2, target_atoms=1000)
atoms = [glass_Atoms(atom) for atom in atoms]
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
The `diffusion` class contains functions for calculating diffusion properties of materials, as averaged over a list of Extended ASE Atoms objects.

```
from vitrum.scattering import diffusion
diffusion_funcs = diffusion(atoms)
```

To calculate the mean squared displacement of a material, the following line can be used:

```
msd = diffusion_funcs.calculate_mean_square_displacement()
```

To get the timesteps at which the mean squared displacement was calculated, one of the utility functions can be used:

```
from vitrum.utility import get_LAMMPS_dump_timesteps
timesteps = get_LAMMPS_dump_timesteps('md.lammpstrj')
```
