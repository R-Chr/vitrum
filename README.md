# ⏳ vitrum⏳ 

Implementation of various analysis methods commonly used for disordered and glassy material structures.

Package is built as an extension to the ASE python package


## 🎯 Scope and functionality

### Working implementations
| Analysis method | Class method |
| ----------- | ----------- |
| Partial Radial distribution functions | .get_rdf() |
| Total radial distribution function | get_total_rdf() | 
| Angle distribution function | .get_angular_dist() |
| Coordination number | .get_coordination_number() |
| Qn analysis | .get_qn() |
| Persistence diagram | .get_persistence_diagram() |
| Local persistence | .get_local_persistence() |

#### Additional utility functions:

| Utility | Function |
| ----------- | ----------- |
| Get sampled timesteps in a LAMMPS dump file | get_LAMMPS_dump_timesteps() |

### Possible future methods:
#### Structure

Structure factor
Ring analysis

#### Dynamics methods:
Mean square displacement
Van hove correlation functions
Velocity Auto Correlation function


## 📖 Author
Author: Rasmus Christensen (rasmusc@bio.aau.dk)

## ⭐ Acknowledgements

`vitrum` has been built with the help of several open-source packages.
All of these are listed in setup.py.

These packages include:
ASE
Numpy
Pandas
Dionysus
Diode
