<div class="vitrum-hero">
  <img alt="vitrum, glass structure analysis" src="vitrum.png#only-light">
  <img alt="vitrum, glass structure analysis" src="vitrum_light.png#only-dark">
</div>

**vitrum** reads a glass structure or an MD trajectory and returns the numbers usually quoted from one: RDFs and structure factors, coordination numbers and Q^n speciation, ring and void statistics, persistence diagrams, diffusion coefficients. It also builds the structures in the first place, by random packing or by sampling a composition space.

## Documentation
The navigation on the left covers each module. Source code is in the [GitHub repository](https://github.com/R-Chr/vitrum).

## Installation
See [Installation](vitrum/install.md) for full instructions, including optional dependency groups.

## Active development
vitrum is under active development. Before 2.0, a minor release may still remove or rename API that turned out to be wrong. Every such change is listed in the [changelog](https://github.com/R-Chr/vitrum/blob/main/CHANGELOG.md), and anything scheduled for removal is deprecated with a warning naming its replacement first where practical. From 2.0 onwards the public API follows [semantic versioning](https://semver.org/).

## Examples
The [`examples`](https://github.com/R-Chr/vitrum/tree/main/examples) folder on GitHub holds runnable notebooks: scattering and RDF analysis, Q^n speciation, random structure generation, and more.

## What it computes

### Structure
*   Partial and total radial distribution functions and structure factors $S(q)$, weighted for neutron or X-ray scattering (`vitrum.scattering`).
*   Ring size distributions and per-ring topology metrics in network glasses (`vitrum.rings`).
*   Free volume fraction and discrete cavity sizes, from a probe-accessible grid (`vitrum.voids`).
*   Persistence diagrams, for medium-range order and topological features (`vitrum.persistent_homology`).
*   Bond angle distributions, coordination numbers and Q^n speciation (`vitrum.coordination`).

### Dynamics
*   Mean squared displacement, diffusion coefficients and Van Hove correlation functions (`vitrum.diffusion`).

### Machine learning and workflows
*   The BALACE framework (`vitrum.batch_active`, needs the `workflows` extra) trains ACE interatomic potentials by batch active learning, running VASP and LAMMPS through Fireworks and Jobflow. It is stale and unsupported, see [Known issues](vitrum/known_issues.md).


## Citation
If you use `vitrum` in your work, please cite it. Each GitHub release is archived on Zenodo with a version-specific DOI; see [`CITATION.cff`](https://github.com/R-Chr/vitrum/blob/main/CITATION.cff) in the repository for the citation metadata.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21366368.svg)](https://doi.org/10.5281/zenodo.21366368)

## Author
Rasmus Christensen (rasmusc@bio.aau.dk)

## Acknowledgements
`vitrum` builds on:
*   [ASE](https://wiki.fysik.dtu.dk/ase/)
*   [Pymatgen](https://pymatgen.org/)
*   [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) / [pandas](https://pandas.pydata.org/)
*   [Dionysus](https://mrzv.org/software/dionysus2/) / [DioDe](https://github.com/mrzv/diode)
*   [Atomate2](https://github.com/materialsproject/atomate2) / [Jobflow](https://materialsproject.github.io/jobflow/) / [Fireworks](https://materialsproject.github.io/fireworks/)
