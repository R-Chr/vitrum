## Installation

### (optional) A conda environment
A separate environment keeps vitrum's dependencies out of the way of everything else:
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

### (optional) batch_active (the BALACE framework)
```
pip install "vitrum[workflows]"
```
This pulls in FireWorks, jobflow and atomate2, plus the YAML and scikit-learn packages
`batch_active` needs on top of them. Note that `batch_active` is unsupported, see
[Known issues](known_issues.md).

### (optional) Materials Project volume lookups
```
pip install "vitrum[volume_estimation]"
```
`vitrum.volume_estimation` needs this for its `"mp"`, `"icsd"` and `"convex_hull"` volume
sources, which `get_random_packed` calls into.

### (optional) Dionysus and Diode
Persistent homology needs both of these.
```
pip install "vitrum[persistent_homology]"
pip install git+https://github.com/mrzv/diode.git
```

DioDe uses [CGAL](http://www.cgal.org/) to generate alpha shapes filtrations in a format that Dionysus understands. DioDe therefore needs CGAL installed as well.

### (optional) Plotly and OVITO
For the interactive 3D void visualization, `VoidAnalysis.plot_3d` and rendering structure images/widgets via `vitrum.visualization.StructureRenderer` (uses [OVITO](https://www.ovito.org/) for Tachyon rendering):

```
pip install "vitrum[visualization]"
```