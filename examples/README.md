# Examples

## `analysis/demo.ipynb`

Demonstrates the structural-characterization workflow on an existing LAMMPS MD trajectory: loading a dump file, correcting chemical symbols, computing the neutron structure factor S(Q) and partial pair distribution function g(r) with `vitrum.scattering.Scattering`, Qn speciation via `GlassAtoms.get_bridging_analysis`, and generating a random packed structure with `vitrum.packing.get_random_packed`.

Requires `md.lammpstrj`, which is included alongside the notebook in this folder.

## `ace-potential/demo.ipynb`

Demonstrates the BALACE (Batch Active Learning for ACE potentials) workflow via `vitrum.batch_active`. Requires a configured LAMMPS (with ML-PACE), VASP, FireWorks/MongoDB, and pacemaker/pyace environment — see [docs/vitrum/balace.md](../docs/vitrum/balace.md) for setup instructions.
