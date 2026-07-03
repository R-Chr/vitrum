# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0]

First stable release, published to PyPI.

### Changed
- Migrated packaging from `setup.py` to `pyproject.toml` (PEP 621); version is now
  sourced from `vitrum.__version__` as the single source of truth.
- Raised the minimum supported Python version to 3.10.
- Split the former `vitrum.utility` grab-bag module into focused modules:
  `geometry`, `trajectory`, `packing`, `volume_estimation`, `io_helpers`,
  `structure_validation`, `comparison`.
- Renamed `glass_Atoms.py` to `glass_atoms.py` and removed the redundant
  lowercase class aliases (`glass_Atoms`, `coordination`, `diffusion`,
  `scattering`, `RINGs`) in favor of their PascalCase names
  (`GlassAtoms`, `Coordination`, `Diffusion`, `Scattering`, `RingAnalysis`).
- `vitrum/__init__.py` now exports the five main classes directly
  (`from vitrum import GlassAtoms`, etc.).


 