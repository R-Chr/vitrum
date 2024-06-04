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
| Change chemical symbols of glass_Atoms object | .set_new_chemical_symbols( )| 
| Get sampled timesteps in a LAMMPS dump file | get_LAMMPS_dump_timesteps( ) |

### Possible future methods:

#### Dynamics methods:
Van hove correlation functions
Velocity Auto Correlation function


## 📖 Author
Author: Rasmus Christensen (rasmusc@bio.aau.dk)

## ⭐ Acknowledgements

`vitrum` has been built with the help of several open-source packages.
All of these are listed in setup.py.

These packages include:
`ASE`
`NumPy`
`sklearn`
`scipy`
`pandas`
`Dionysus`
`Diode`
