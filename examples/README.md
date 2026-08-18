# Examples

## `analysis/demo.ipynb`

Demonstrates the structural-characterization workflow on an existing LAMMPS MD trajectory: loading a dump file, correcting chemical symbols, computing the neutron and x-ray structure factors S(Q) and partial pair distribution functions g(r) with `vitrum.scattering.Scattering`, coordination numbers and bond angle distributions with `vitrum.coordination.Coordination`, ring statistics with `vitrum.rings.RingAnalysis`, cavity analysis with `vitrum.voids.VoidAnalysis`, persistent homology with `vitrum.persistent_homology.PersistenceDiagram`, and generating a random packed structure with `vitrum.packing.get_random_packed`.

Requires `md.lammpstrj`, which is included alongside the notebook in this folder.

## `ace-potential/demo.ipynb`

Demonstrates the BALACE (Batch Active Learning for ACE potentials) workflow via `vitrum.batch_active`. Requires a configured LAMMPS (with ML-PACE), VASP, FireWorks/MongoDB, and pacemaker/pyace environment — see [docs/vitrum/balace.md](../docs/vitrum/balace.md) for setup instructions.
