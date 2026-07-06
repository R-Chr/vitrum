# Persistent homology

Tools for computing persistence diagrams of atomic structures, used to identify medium-range order and topological features (loops/rings via dimension 1, cavities/voids via dimension 2).

`PersistenceDiagram` builds a single weighted alpha-shape filtration from the atoms, with atom weightings (covalent radii by default, consistent with [`vitrum.voids`](voids.md)) controlling how the filtration grows around each atom. Pass `exclude_atoms` to drop chemical species (e.g. network modifiers) from the structure before anything is computed. `.calculate()` computes persistence diagrams for any requested homology dimensions at once, `.get_diagram()` retrieves one, `.get_apf()` / `.plot_apf()` summarize a diagram as an accumulated persistence function (Biscio & Møller, *J. Comput. Graph. Stat.* 2019, 28, 671), `.get_sph()` / `.plot_sph()` compute the S<sub>PH</sub>(Q) function for comparing diagram features directly to a structure factor (Sørensen et al., *Sci. Adv.* 2020, 6, eabc2320), and `.get_persistence_image()` / `.plot_persistence_image()` vectorize a diagram into a fixed-size 2D array suitable for ML features or direct comparison across structures, and `.get_cycle_atoms()` / `.get_diagram_composition()` resolve which atoms make up a given loop's or void's representative cycle, annotating the diagram with cycle size and per-species composition.

Requires the optional `dionysus`/`diode` dependencies — see [Installation](install.md).

::: vitrum.persistent_homology
