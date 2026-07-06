---
title: 'vitrum: A Python package for the generation, analysis, and simulation of glassy atomic structures'
tags:
  - Python
  - materials science
  - glass
  - amorphous materials
  - molecular dynamics
authors:
  - name: Rasmus Christensen
    orcid: 0000-0003-2654-1545
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Applied Physics, Tohoku University, Japan
    index: 1
date: 3 July 2026
bibliography: paper.bib
---

# Summary

Glasses and other disordered materials lack the long-range periodic order of crystals, which makes their computational study uniquely challenging. Their atomic structure cannot be reduced to a small unit cell, and meaningful characterization requires statistical descriptors computed over large atomistic configurations and long molecular dynamics (MD) trajectories. `vitrum` is a Python package for the generation, analysis, and simulation of disordered and glassy atomic structures. It provides a coherent suite of tools covering typical simulation workflow and analysis for glassy materials: structural characterization, including partial and total radial distribution functions (RDFs), neutron- and X-ray-weighted structure factors $S(q)$, coordination environments, bond-angle distributions, ring-size statistics of network glasses, and persistent homology for quantifying medium-range order.

`vitrum` builds on established libraries in the atomistic simulation ecosystem, including the Atomic Simulation Environment (ASE) [@Larsen2017], `pymatgen` [@Ong2013], NumPy [@Harris2020], SciPy [@Virtanen2020], pandas [@McKinney2010], and scikit-learn [@Pedregosa2011]. Structures are handled through an ASE `Atoms` interface, so that results from any MD engine or structure generator readable by ASE can be analyzed directly. Persistent homology computations use Dionysus and DioDe [@Morozov2016]. The package is MIT licensed, documented at https://vitrum.readthedocs.io, and distributed with runnable example notebooks covering scattering analysis, $Q^n$ speciation, and random structure generation.

# Statement of need

The atomistic modeling of oxide, chalcogenide, and metallic glasses is a large and growing field, driven both by classical MD and, increasingly, by machine learning interatomic potentials that offer near-ab-initio accuracy at a fraction of the cost. The analysis tooling in this field, however, remains fragmented, and none of the widely used packages serves glass science well on its own. `pymatgen` [@Ong2013] is a comprehensive materials analysis library, but its data structures and analysis modules are built around crystallographic symmetry, space groups, and periodic unit cells — concepts that have no meaningful counterpart in disordered structures. ASE [@Larsen2017] is structure-agnostic and excels as a low-level framework for building, running, and reading atomistic simulations across many MD and DFT codes, but it deliberately remains a framework and offers little in the way of higher-level structural analysis. General-purpose trajectory analysis packages such as `MDAnalysis` [@MichaudAgrawal2011] and `freud` [@Ramasubramani2020] supply RDFs and some order parameters, but not the descriptors that are the actual currency of glass structure analysis: experiment-comparable neutron- and X-ray-weighted total structure factors, ring statistics of network formers, $Q^n$ speciation, or persistent-homology-based measures of medium-range order [@Hiraoka2016]. Specialized tools do exist for individual tasks — e.g., R.I.N.G.S. [@LeRoux2010] for ring analysis or standalone persistent homology pipelines — but they are distributed as separate codes with heterogeneous input formats, forcing researchers to maintain brittle conversion scripts between them.

`vitrum` fills this gap by building directly on ASE's `Atoms` object — inheriting its generality and interoperability with the wider simulation ecosystem — while supplying the glass-specific analysis layer that existing packages lack, all behind a single, unified data model. This reduces the incidental complexity of computational glass science and makes analyses reproducible without ad hoc glue code. The target audience is researchers in glass and amorphous materials science, from students running their first MD simulations of silicate glasses to groups studying novel multicomponent systems. `vitrum` has been developed to support ongoing research on disordered materials and is intended as a community resource for reproducible glass simulation workflows.

# Functionality

The main capabilities of `vitrum` are organized into focused modules:

- `vitrum.scattering`: partial and total RDFs and structure factors $S(q)$ with neutron and X-ray weighting for direct comparison to diffraction experiments.
- `vitrum.coordination`: coordination number statistics, bond-angle distributions, and speciation analysis of local environments.
- `vitrum.rings`: ring-size distributions and statistics for network glasses.
- `vitrum.persistent_homology`: persistence diagrams for characterizing medium-range order and topological features of disordered networks.

All analyses operate on an extended ASE `Atoms` object, making the package interoperable with the wider ASE/`pymatgen` ecosystem and agnostic to the simulation engine used to produce the data.

# Acknowledgements

`vitrum` relies on several open-source packages, including ASE, `pymatgen`, NumPy, SciPy, pandas, scikit-learn, Dionysus, and DioDe.

# References