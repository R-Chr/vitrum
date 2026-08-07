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
date: 7 August 2026
bibliography: paper.bib
---

# Summary

Glasses and other disordered materials lack the long-range periodic order of crystals, making analysis of their atomic structure inherently different. A glass is specified by no finite set of coordinates, only by the statistics of the local atomic environments present. Characterizing these materials therefore requires computing statistical descriptors over configurations of thousands of atoms, and often averaging them over several frames from molecular dynamics trajectories. The structural descriptors of interest for amorphous materials often include: short-range coordination environments and bond angles, shortest-path ring statistics, the free volume between atoms, topology of the atomic point cloud, and the set of scattering functions which can be compared against experiment. `vitrum` is a Python package implementing these descriptors, together with other tooling, such as composition sampling and structure packing needed for simulating amorphous materials. It is built on the Atomic Simulation Environment [@ase], ensuring interoperability with the wider atomistic simulation ecosystem. This paper describes `vitrum` v1.1.0; the package continues to develop, and the current interface and its limitations are documented with the package.


# Statement of need

Atomistic modeling of oxide, chalcogenide, and metallic glasses is driven by classical molecular dynamics and, increasingly, by machine learning interatomic potentials. The analysis tooling remains fragmented, and no widely used package serves glass and amorphous science well on its own. `pymatgen` [@pymatgen] is organized around crystallographic symmetry, space groups, and periodic unit cells, concepts that have no meaningful counterpart in a disordered solid. ASE [@ase] is structure-agnostic and works well as a low-level framework for building, running, and reading atomistic simulations, but by design it offers little higher-level analysis. General-purpose trajectory packages such as `MDAnalysis` [@mdanalysis] and `freud` [@freud] are built around radial distribution functions and order parameters developed for soft matter, rather than the descriptors typically reported for glass structure: experiment-comparable neutron- and X-ray-weighted total structure factors, ring statistics of network formers, $Q^n$ speciation, or topological measures of medium-range order. Specialized tools cover parts of these tasks, such as R.I.N.G.S. [@leroux2010] for ring analysis, but they are separate codes with heterogeneous input formats, and do not always integrate well with modern workflows. At the time of writing, we are not aware of a package that covers this specific combination of requirements.

The design follows from that of ASE, whose central object is the `Atoms` class. `vitrum` takes plain `Atoms` objects throughout, so a structure or trajectory read by ASE is passed directly to an analysis with no conversion step. Each analysis is a thin class constructed from a single configuration or a list of frames, currently `Scattering`, `Coordination`, `RingAnalysis`, `VoidAnalysis`, and `PersistenceDiagram`, which between them cover the per-structure and trajectory-averaged queries meaningful for a glass: partial pair distribution functions, coordination numbers, bond angles, bridging-atom speciation, ring statistics, cavity volumes, and persistence diagrams. Each exposes a small set of `get_*` methods returning plain NumPy arrays [@numpy] and pandas DataFrames, so output flows directly into the user's own fitting and plotting. Any trajectory ASE can read is therefore analyzable, independent of the engine that produced it.

# Functionality

The main capabilities of `vitrum` are organized into focused modules:

- **Scattering and short-range order.** Diffraction is one of the primary experimental probes of glass structure, and simulated structures are commonly validated by their agreement with the measured correlation functions. `vitrum.scattering.Scattering` computes partial and total pair distribution functions $g(r)$, the total correlation function $T(r)$, the reduced pair distribution function $G(r)$, partial and total structure factors $S(Q)$, and running coordination numbers $N(r)$, following the definitions collected by @keen2001. Faber–Ziman neutron weighting [@sears1992], $Q$-dependent X-ray weighting from Cromer–Mann coefficients [@intltables], and a $Q$-independent atomic-number approximation are available, and finite-resolution effects are reproduced through the Lorch modification function [@lorch1969] and Gaussian broadening at a chosen $Q_{max}$. `vitrum.comparison.r_chi` quantifies agreement with a measured function through the Wright $R_\chi$ goodness-of-fit factor [@wright1993]. `vitrum.coordination.Coordination` aggregates coordination number, bond-angle, and $Q^n$ speciation distributions over a trajectory.
- **Ring statistics.** Beyond the first coordination shell, the connectivity of a network glass is conventionally characterized by shortest-path ring statistics, which distinguish structures identical in their pair correlations. `vitrum.rings` implements the Guttman [@guttman1990], King [@king1967], and primitive [@franzblau1991] criteria following the definitions used by the R.I.N.G.S. code [@leroux2010]. Rings are returned as objects, not counts alone, carrying shape descriptors such as perimeter, area, radius of gyration, roundness, and planeness.
- **Free volume and cavities.** `vitrum.voids.VoidAnalysis` labels the connected cavities in the space left outside the atoms' exclusion spheres, so that a cavity straddling a cell face is counted once. It reports the free volume fraction, each cavity's volume, effective radius, void centre, and size distributions over the cell. Cavity centres can be written out as pseudo-atoms for viewing in a standard visualizer, and the free-space field rendered as an isosurface.
- **Topological data analysis.** Persistent homology has become an established descriptor for capturing the medium-range order in disordered solids [@hiraoka2016; @sorensen2020]. `vitrum.persistent_homology.PersistenceDiagram` includes methods for building weighted alpha-shape filtrations, currently through DioDe and Dionysus [@dionysus], and returns persistence diagrams in any homology dimension, most usefully $H_1$ and $H_2$. It also resolves the representative cycle of a feature back to the atoms that form it and reports that cycle's composition. Diagrams are summarized as accumulated persistence functions [@biscio2019], as the $S_{PH}(Q)$ function that maps topological features onto the same axis as a measured structure factor [@sorensen2020], and as persistence images [@adams2017] for machine learning.
- **Structure generation.** Glass properties are often studied based on how they vary across a compositional space, so a study often needs many candidate structures rather than one. `vitrum.structure_gen.GlassGenerator` samples composition space as mole fractions of oxide units or as charge-neutral elemental fractions, using Sobol or Latin hypercube sequences stratified over binary, ternary, and higher-order subsystems. `vitrum.packing.get_random_packed` turns a given composition into a periodic simulation cell, sizing the cell from ionic or covalent radii, from a specified density, or from Materials Project [@mp2013] convex-hull queries.

![Structural analysis of a 3000-atom 30Na$_2$O–70SiO$_2$ sodium silicate glass, computed with `vitrum` from the molecular dynamics trajectory shipped with the package. (a) Total structure factor under neutron and $Q$-dependent X-ray weighting. (b) Intra- and inter-tetrahedral bond angle distributions, O–Si–O and Si–O–Si. (c) Ring size distribution of the Si–O network in the final frame under the Guttman and King criteria, normalized by cell volume. The figure is reproduced by `paper/make_figure.py`.\label{fig:overview}](figure1.png)

# General-purpose tools

`vitrum` also provides the smaller utilities a glass simulation workflow keeps needing. `vitrum.io_helpers` remaps the numeric atom types of a LAMMPS dump [@lammps] onto chemical symbols, converts between mass and number density, and converts a mol% oxide specification such as `60SiO2-25Na2O-15CaO` into a single formula unit using exact rational arithmetic. `vitrum.visualization.StructureRenderer` wraps OVITO [@ovito] for ray-traced renderings coloured by species or by any scalar property. Trajectory unwrapping and mean-square-displacement analysis are available, though for quantitative diffusion work we recommend dedicated packages such as `kinisi` [@kinisi] and GEMDAT [@gemdat] instead.

# Quality assurance

`vitrum` is covered by a test suite built on analytically constructed structures rather than stored data files, so that every expected value is known independently of this code's own output. Ring counts, coordination numbers, bond angles, and densities are checked against crystalline reference structures, primitive ring counts against brute-force enumeration of the bond graph, and the scattering routines against analytic limits and internal consistency requirements. Continuous integration runs the suite on all currently supported Python versions, and current limitations are documented in the package.

# Acknowledgements

`vitrum` builds on ASE [@ase], pymatgen [@pymatgen], NumPy [@numpy], SciPy [@scipy], pandas [@pandas], scikit-learn [@sklearn], Numba [@numba], Matplotlib [@matplotlib], Dionysus and DioDe [@dionysus], and OVITO [@ovito]. The ring analysis implementation is adapted from `sova-cui` (<https://github.com/MotokiShiga/sova-cui>), and the approach to three-dimensional void rendering is adapted from GEMDAT [@gemdat].

<!-- TODO: add funding sources and grant numbers before submission. -->

# References