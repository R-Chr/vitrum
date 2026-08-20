<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/vitrum_light.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/vitrum.png">
    <img alt="vitrum, glass structure analysis" src="docs/vitrum.png" width="520">
  </picture>
</p>

<p align="center">
  <a href="https://vitrum.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/vitrum/badge/?version=latest"></a>
  <a href="https://pypi.org/project/vitrum/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/vitrum"></a>
  <a href="https://pypi.org/project/vitrum/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vitrum.svg?style=flat"></a>
</p>

**vitrum** reads a glass structure or an MD trajectory and returns the numbers usually quoted from one: RDFs and structure factors, coordination numbers and Q^n speciation, ring and void statistics, persistence diagrams, diffusion coefficients. It also builds the structures in the first place, by random packing or by sampling a composition space.

## 🚧 Active development
vitrum is under active development. Before 2.0, a minor release may still remove or rename API that turned out to be wrong. Every such change is listed in the [changelog](CHANGELOG.md), and anything scheduled for removal is deprecated with a warning naming its replacement first where practical. From 2.0 onwards the public API follows [semantic versioning](https://semver.org/).

## 📖 Documentation
The `docs` folder holds the sources. The built site is at [vitrum.readthedocs.io](https://vitrum.readthedocs.io/en/latest/).

## 📦 Installation

`vitrum` is available on [PyPI](https://pypi.org/project/vitrum/):

```bash
pip install vitrum
```

To install dependencies for simulation workflows (atomate2, fireworks, jobflow):

```bash
pip install vitrum[workflows]
```

For the latest development version, clone the repository and install it in editable mode instead:

```bash
git clone https://github.com/R-Chr/vitrum.git
cd vitrum
pip install -e .
```

## 🚀 Examples
The [`examples`](examples/) folder holds runnable notebooks: scattering and RDF analysis, Q^n speciation, random structure generation, and more.

## 🎯 What it computes

### Structure
*   Partial and total radial distribution functions and structure factors $S(q)$, weighted for neutron or X-ray scattering (`vitrum.scattering`).
*   Ring size distributions and per-ring topology metrics in network glasses (`vitrum.rings`).
*   Free volume fraction and discrete cavity sizes, from a probe-accessible grid (`vitrum.voids`).
*   Persistence diagrams, for medium-range order and topological features (`vitrum.persistent_homology`).
*   Bond angle distributions, coordination numbers and Q^n speciation (`vitrum.coordination`).

### Dynamics
*   Mean squared displacement, diffusion coefficients and Van Hove correlation functions (`vitrum.diffusion`).


## 📑 Citation
If you use `vitrum` in your work, please cite it. Each GitHub release is archived on Zenodo with a version-specific DOI; see [`CITATION.cff`](CITATION.cff) for the citation metadata (GitHub's "Cite this repository" button uses this file automatically).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21366368.svg)](https://doi.org/10.5281/zenodo.21366368)


## 🤝 Contributing
Bug reports, test cases and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the development setup and what a mergeable change looks like, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Security issues should
go through [SECURITY.md](SECURITY.md) rather than the public issue tracker.

## 👥 Author
Rasmus Christensen (rasmus.christensen.a1@tohoku.ac.jp)

## ⭐ Acknowledgements
`vitrum` builds on:
*   [ASE](https://wiki.fysik.dtu.dk/ase/)
*   [Pymatgen](https://pymatgen.org/)
*   [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) / [pandas](https://pandas.pydata.org/)
*   [scikit-learn](https://scikit-learn.org/)
*   [Dionysus](https://mrzv.org/software/dionysus2/) / [DioDe](https://github.com/mrzv/diode)
*   [Atomate2](https://github.com/materialsproject/atomate2) / [Jobflow](https://materialsproject.github.io/jobflow/) / [Fireworks](https://materialsproject.github.io/fireworks/)