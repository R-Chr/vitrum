# ⏳ vitrum⏳ 

Implementation of various analysis methods commonly used for disordered and glassy material structures.

Package is built as an extension to the ASE python package


## 🎯 Scope and functionality

### Working implementations
#### Classes
| Class name | Functionality |
| ----------- | ----------- |
| glass_Atoms | Extension to the ASE Atoms object, implementing various analysis of individual structures|
| scattering | Calculate scattering functions based on a list of glass_Atoms objects|
| diffusion | Calculate diffusion properties based on a trajectory of glass_Atoms objects |
| persistent_homology | Calculate persistent homology based on a list of glass_Atoms objects |


#### glass_Atoms
| Analysis method | Class method |
| ----------- | ----------- |
| Partial Radial distribution functions | .get_rdf() |
| Total radial distribution function | get_total_rdf() | 
| Faber-Ziman partial structure factor | .get_partial_structure_factor() | 
| Structure factor | .get_strucutre_factor() | 
| Angle distribution function | .get_angular_dist() |
| Coordination number | .get_coordination_number() |
| Qn analysis | .get_qn() |
| Persistence diagram | .get_persistence_diagram() |
| Local persistence | .get_local_persistence() |

#### scattering
| Analysis method | Class method |
| ----------- | ----------- |
| Partial Radial distribution functions | .get_partial_pdf( ) |
| Total radial distribution function | get_total_rdf( ) | 
| Faber-Ziman partial structure factor | .get_partial_structure_factor( ) | 
| Structure factor | .get_strucutre_factor( ) | 

#### diffusion
| Analysis method | Class method |
| ----------- | ----------- |
| Mean square displacement | .get_mean_square_displacement( ) |



#### Additional utility functions:

| Utility | Function |
| ----------- | ----------- |
| Generate random structure of a given compositon| get_random_packed()| 
| Get sampled timesteps in a LAMMPS dump file | get_LAMMPS_dump_timesteps( ) |

### Possible future methods:

#### Dynamics methods:
Van hove correlation functions
Velocity Auto Correlation function

## ⚙️ Installation

### (optional) Creating a conda environment
It is common practice creating a separate conda environment to avoid dependencies mixing. You can create the new environment named vitrum with minimal amount of required packages with the following command:
```
conda create -n vitrum python=3.11
conda activate vitrum
```
### Installation of vitrum
To install vitrum:

pip install directly  from this repository.
```
pip install git+https://github.com/R-Chr/vitrum.git
```

To update package to the most current version
```
pip install --force-reinstall --no-deps git+https://github.com/R-Chr/vitrum.git
```

### Dionysus and Diode
For persistent homology analsysis these packages are required. They are however currently required installs to avoid errors, may change in the future.
```
pip install dionysus
pip install git+https://github.com/mrzv/diode.git
```
### (optional) CGAL for Diode
DioDe uses [CGAL](http://www.cgal.org/) to generate alpha shapes filtrations in a format that Dionysus understands. For DioDe to work [CGAL](http://www.cgal.org/) is required (Only important for persistent homology).


## 📖 Author
Author: Rasmus Christensen (rasmusc@bio.aau.dk)

## ⭐ Acknowledgements

`vitrum` has been built with the help of several open-source packages. All of these are listed in setup.py.

These packages include:
[`ASE`](https://wiki.fysik.dtu.dk/ase/index.html)
[`NumPy`](https://numpy.org/)
[`scikit-learn`](https://scikit-learn.org/stable/)
[`scipy`](https://scipy.org/)
[`pandas`](https://pandas.pydata.org/)
[`Dionysus`](https://mrzv.org/software/dionysus2/)
[`DioDe`](https://github.com/mrzv/diode)
[`pymatgen`](https://pymatgen.org/)