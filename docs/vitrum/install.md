## ⚙️ Installation

### (optional) Creating a conda environment
It is common practice creating a separate conda environment to avoid dependencies mixing. You can create the new environment named vitrum with minimal amount of required packages with the following command:
```
conda create -n vitrum python=3.11
conda activate vitrum
```
### Installation of vitrum
To install vitrum:

pip install directly  from this repository. (Make sure you have git installed)
```
pip install git+https://github.com/R-Chr/vitrum.git
```

To update package to the most current version
```
pip install --force-reinstall --no-deps git+https://github.com/R-Chr/vitrum.git
```

### (optional) Dionysus and Diode
For persistent homology analsysis these packages are required.
```
pip install dionysus
pip install git+https://github.com/mrzv/diode.git
```

DioDe uses [CGAL](http://www.cgal.org/) to generate alpha shapes filtrations in a format that Dionysus understands. For DioDe to work [CGAL](http://www.cgal.org/) is required (Only important for persistent homology).