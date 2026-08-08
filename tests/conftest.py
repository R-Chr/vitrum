"""Shared fixtures for the vitrum test suite.

Every expected value is known independently of this code: from crystallography for the
generated structures, and from stoichiometry for the one real trajectory. Nothing here is
a number recorded from vitrum's own output.

The crystals are for the exact answers -- a coordination number that is 4 and nothing
else, a ring count fixed by symmetry, an asymmetric 8-vs-4 fluorite. The glass is for
everything that only means something in a disordered network: PDF and S(Q) asymptotes,
angle distributions, Qn speciation, diffusion.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read

from vitrum.io_helpers import correct_atom_types

# Lattice constant of crystalline silicon (diamond structure), in Angstrom.
SI_LATTICE_CONSTANT = 5.431
# Nearest-neighbour distance in the diamond structure: a * sqrt(3) / 4.
SI_NN_DISTANCE = SI_LATTICE_CONSTANT * np.sqrt(3) / 4
# Between the 1st (2.3517 A) and 2nd (3.840 A) neighbour shells in diamond Si.
SI_FIRST_SHELL_CUTOFF = 3.0

# Lattice constant of fluorite CaF2, in Angstrom.
CAF2_LATTICE_CONSTANT = 5.451
# Between the 1st (2.36 A) and 2nd (2.73 A) shells in fluorite CaF2.
CAF2_FIRST_SHELL_CUTOFF = 2.5

# --- the example glass trajectory ---------------------------------------------------------
#
# examples/analysis/md.lammpstrj: 100 frames of 3000 atoms, a 34.44 A cubic cell at
# 2.47 g/cm^3. LAMMPS types map to species as in docs/vitrum/quickstart.md.
TRAJECTORY_PATH = Path(__file__).resolve().parents[1] / "examples" / "analysis" / "md.lammpstrj"
LAMMPS_TYPE_TO_SYMBOL = {1: "Na", 2: "O", 3: "Si"}

# Composition of that trajectory: Na600 O1700 Si700, i.e. 70 SiO2 . 30 Na2O.
N_NA, N_O, N_SI = 600, 1700, 700
# Every Na contributes one non-bridging oxygen, so the rest of the oxygens bridge.
BRIDGING_OXYGEN_FRACTION = 1 - N_NA / N_O
# Each non-bridging oxygen costs one bridge from a tetrahedron: <Qn> = 4 - NBO per Si.
MEAN_Q_SPECIES = 4 - N_NA / N_SI
# Comfortably past the 1.60 A Si-O bond and short of any second shell.
SI_O_CUTOFF = 2.0


def _read_glass(index):
    atoms = read(TRAJECTORY_PATH, index=index, format="lammps-dump-text")
    correct_atom_types(atoms, LAMMPS_TYPE_TO_SYMBOL)
    return atoms


@pytest.fixture(scope="session")
def silicon_diamond():
    """Crystalline silicon, 3x3x3 of the cubic 8-atom cell (216 atoms).

    Ground truth: every atom has exactly 4 nearest neighbours at SI_NN_DISTANCE,
    and the shortest rings are 6-membered.
    """
    return bulk("Si", "diamond", a=SI_LATTICE_CONSTANT, cubic=True).repeat(3)


@pytest.fixture(scope="session")
def silicon_small():
    """Crystalline silicon, 2x2x2 of the cubic 8-atom cell (64 atoms).

    Same ground truth as silicon_diamond, but small enough for ring enumeration.
    """
    return bulk("Si", "diamond", a=SI_LATTICE_CONSTANT, cubic=True).repeat(2)


@pytest.fixture(scope="session")
def fluorite_caf2():
    """Fluorite CaF2, 3x3x3 of the cubic cell (324 atoms).

    Ground truth from the fluorite structure: every Ca has 8 F neighbours at
    a*sqrt(3)/4 = 2.36 A, every F has 4 Ca neighbours at the same distance, and
    every F has 6 F neighbours at a/2 = 2.73 A. The 8 vs 4 asymmetry is what makes
    this a good test of a coordination number that depends on which species is the
    centre and which is the neighbour.
    """
    from ase.spacegroup import crystal

    a = CAF2_LATTICE_CONSTANT
    return crystal(
        ["Ca", "F"],
        basis=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
        spacegroup=225,
        cellpar=[a, a, a, 90, 90, 90],
        size=(3, 3, 3),
    )


@pytest.fixture(scope="session")
def sodium_silicate():
    """Five frames of the 30Na2O-70SiO2 melt, every 20th of the 100 in the trajectory.

    A real disordered network, so the quantities that only exist in one -- a spread of
    bond angles, a Qn distribution, a PDF that decays to 1 -- are measured on the thing
    they describe. The expected values come from the composition (see the constants
    above) and from silicate chemistry, not from vitrum.
    """
    return _read_glass("::20")


@pytest.fixture(scope="session")
def sodium_silicate_frame(sodium_silicate):
    """A single frame of the melt, for the single-structure code paths."""
    return sodium_silicate[0]


@pytest.fixture(scope="session")
def sodium_silicate_full():
    """All 100 frames, 1 ps apart. Only diffusion needs the time axis."""
    return _read_glass(":")


@pytest.fixture(scope="session")
def silicon_cubic_cell():
    """Crystalline silicon, a single cubic 8-atom cell.

    Small enough that the minimum-image bond graph contains 4-cycles that wind around
    the cell, so the ring search has to look past them to reach the real 6-rings.
    """
    return bulk("Si", "diamond", a=SI_LATTICE_CONSTANT, cubic=True)


@pytest.fixture(scope="session")
def simple_cubic():
    """A simple-cubic lattice, 3x3x3 of the one-atom cell (27 atoms).

    Ground truth from brute-force enumeration of every non-wrapping cycle of six atoms
    or fewer, filtered by Franzblau's criterion: 81 primitive 4-rings and 108 primitive
    6-rings. The 6-rings are the ones that run around a lattice cube, and every atom on
    one has a two-hop shortcut between its two ring neighbours, so no ring criterion
    based on shortest paths through a single atom can find them.
    """
    return bulk("Po", "sc", a=3.0).repeat(3)


@pytest.fixture(scope="session")
def cube_graph():
    """Eight atoms on the corners of a 2.2 A cube, isolated (no periodicity).

    The bond graph is the cube graph Q3: edges bond (2.2 A), face diagonals do not
    (3.11 A, against a 2.89 A cutoff). Ground truth by brute force: 6 primitive 4-rings
    (the faces) and 4 primitive 6-rings.
    """
    positions = [(x, y, z) for x in (0.0, 2.2) for y in (0.0, 2.2) for z in (0.0, 2.2)]
    return Atoms("Si8", positions=positions, cell=[30.0] * 3, pbc=False)


@pytest.fixture
def triclinic_atoms():
    """A structure with a genuinely triclinic cell, for the orthorhombic guard."""
    atoms = bulk("Si", "diamond", a=SI_LATTICE_CONSTANT, cubic=True).repeat(2)
    cell = np.array(atoms.get_cell())
    cell[1, 0] += 2.0  # introduce an off-diagonal component
    atoms.set_cell(cell, scale_atoms=False)
    return atoms


@pytest.fixture
def one_dimer():
    """Six O atoms forming exactly one dimer, none of them across a boundary."""
    positions = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],  # bonded to the atom above (1.0 A apart)
        [10.0, 10.0, 10.0],
        [15.0, 15.0, 15.0],
        [5.0, 15.0, 5.0],
        [15.0, 5.0, 15.0],
    ]
    return Atoms("O6", positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)


@pytest.fixture
def boundary_dimer():
    """Three O atoms; the only dimer is formed across the periodic boundary."""
    positions = [
        [0.5, 1.0, 1.0],
        [19.5, 1.0, 1.0],  # 1.0 A from the atom above, through the boundary
        [10.0, 10.0, 10.0],
    ]
    return Atoms("O3", positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)


@pytest.fixture
def uniform_lattice():
    """A perfectly uniform simple-cubic lattice: unambiguously homogeneous."""
    n = 8
    spacing = 2.5
    positions = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)]) * spacing
    box = n * spacing
    return Atoms(f"Ar{len(positions)}", positions=positions, cell=[box] * 3, pbc=True)


@pytest.fixture
def uniform_random():
    """2000 atoms placed uniformly at random: statistically homogeneous.

    Unlike a crystal lattice this has no periodicity to fall in or out of step with
    the analysis grid, so it stays homogeneous at any grid density.
    """
    rng = np.random.default_rng(3)
    box = 20.0
    positions = rng.random((2000, 3)) * box
    return Atoms("Ar2000", positions=positions, cell=[box] * 3, pbc=True)


@pytest.fixture
def phase_separated():
    """All atoms crammed into one octant of the cell: unambiguously inhomogeneous."""
    rng = np.random.default_rng(7)
    box = 20.0
    positions = rng.random((500, 3)) * (box / 4)
    return Atoms("Ar500", positions=positions, cell=[box] * 3, pbc=True)
