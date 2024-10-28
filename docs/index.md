# ⏳ vitrum⏳ 

Implementation of various analysis methods commonly used for disordered and glassy material structures.

## 🔴 Vitrun is a work in progress 🔴
 Nothing is sacred, and development is ongoing. APIs and functionality are subject to change at any time.



## 🎯 Scope and functionality

#### Classes
| Class name | Functionality |
| ----------- | ----------- |
| glass_Atoms | Extension to the ASE Atoms object, implementing various analysis of individual structures|
| scattering | Calculate scattering functions based on a list of glass_Atoms objects|
| diffusion | Calculate diffusion properties based on a trajectory of glass_Atoms objects |
| persistent_homology | Calculate persistent homology based on a list of glass_Atoms objects |


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