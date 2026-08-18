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


### (optional) A faster neighbour search
```
pip install "vitrum[fast]"
```
Coordination numbers, neighbour lists, bond angles, `Q^n` speciation and partial PDFs are
all reductions of one neighbour search, and that search is where nearly all of their time
goes. This extra installs [matscipy](https://github.com/libAtoms/matscipy), whose C++
implementation is several times faster; `vitrum` picks it up automatically and falls back
to the ASE one when it is absent. Results are unchanged either way.

Note that matscipy is LGPL-2.1 licensed, while `vitrum` itself is MIT. Installing and
importing it alongside `vitrum` is fine, but it is opt-in rather than a core dependency so
that a plain `pip install vitrum` stays permissively licensed throughout.

### (optional) To install dependencies for batch_active (BALACE framework):
```
pip install "vitrum[workflows]"
```
This pulls in `vitrum[workflows]` (FireWorks, jobflow, atomate2) plus the YAML and
scikit-learn packages `batch_active` needs on top of them. Note that
`batch_active` is unsupported — see [Known issues](known_issues.md).

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

DioDe uses [CGAL](http://www.cgal.org/) to generate alpha shapes filtrations in a format that Dionysus understands. For DioDe to work [CGAL](http://www.cgal.org/) is required (Only important for persistent homology).

### (optional) Plotly and OVITO
For the interactive 3D void visualization, `VoidAnalysis.plot_3d` and rendering structure images/widgets via `vitrum.visualization.StructureRenderer` (uses [OVITO](https://www.ovito.org/) for Tachyon rendering):

```
pip install "vitrum[visualization]"
```