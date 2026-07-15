## ⚙️ Installation

### (optional) Creating a conda environment
It is common practice creating a separate conda environment to avoid dependencies mixing. You can create the new environment named vitrum with minimal amount of required packages with the following command:
```
conda create -n vitrum python=3.11
conda activate vitrum
```
### Installation of vitrum
`vitrum` is available on [PyPI](https://pypi.org/project/vitrum/):
```
pip install vitrum
```

To install the latest development version directly from GitHub instead (make sure you have git installed):
```
pip install "vitrum @ git+https://github.com/R-Chr/vitrum.git"
```

To update the development version to the most current commit:
```
pip install --force-reinstall --no-deps "vitrum @ git+https://github.com/R-Chr/vitrum.git"
```


### (optional) To install dependencies for batch_active (BALACE framework):
```
pip install "vitrum[workflows]"
```

### (optional) To install dependencies for Materials Project volume/composition lookups:
```
pip install "vitrum[volume_estimation]"
```
This is required for `vitrum.volume_estimation` (used internally by `get_random_packed`'s `"mp"`/`"icsd"`/`"convex_hull"` volume sources).

### (optional) Dionysus and Diode
For persistent homology analsysis these packages are required.
```
pip install "vitrum[persistent_homology]"
pip install git+https://github.com/mrzv/diode.git
```

DioDe uses [CGAL](http://www.cgal.org/) to generate alpha shapes filtrations in a format that Dionysus understands. For DioDe to work [CGAL](http://www.cgal.org/) is required (Only important for persistent homology). Note: `LocalPD` and `get_local_persistence` in this module are currently non-functional — see [Known Issues](known_issues.md).

### (optional) Plotly and OVITO
For the interactive 3D void visualization, `VoidAnalysis.plot_3d` and rendering structure images/widgets via `vitrum.visualization.StructureRenderer` (uses [OVITO](https://www.ovito.org/) for Tachyon rendering):

```
pip install "vitrum[visualization]"
```