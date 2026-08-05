"""Shared fixtures for the vitrum test suite.

All structures are generated analytically, so the tests need no data files and every
expected value is known from crystallography rather than recorded from this code's own
output.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

# Lattice constant of crystalline silicon (diamond structure), in Angstrom.
SI_LATTICE_CONSTANT = 5.431
# Nearest-neighbour distance in the diamond structure: a * sqrt(3) / 4.
SI_NN_DISTANCE = SI_LATTICE_CONSTANT * np.sqrt(3) / 4
# Second-neighbour distance: a / sqrt(2).
SI_2NN_DISTANCE = SI_LATTICE_CONSTANT / np.sqrt(2)

# Lattice constant of fluorite CaF2, in Angstrom.
CAF2_LATTICE_CONSTANT = 5.451


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
def random_gas():
    """~300 atoms placed at random with a hard-sphere exclusion, in a 24 A cube.

    Ground truth: with no structural correlation beyond the exclusion radius,
    g(r) -> 1 and S(Q) -> 1 at large r / large Q.
    """
    rng = np.random.default_rng(42)
    box = 24.0
    min_sep = 2.2
    points = []
    while len(points) < 300:
        candidate = rng.random(3) * box
        if points:
            delta = np.array(points) - candidate
            delta -= box * np.round(delta / box)  # minimum image
            if np.min(np.linalg.norm(delta, axis=1)) <= min_sep:
                continue
        points.append(candidate)
    points = np.array(points)
    n_si = len(points) // 3
    return Atoms(
        f"Si{n_si}O{len(points) - n_si}",
        positions=points,
        cell=[box, box, box],
        pbc=True,
    )


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


@pytest.fixture(scope="session")
def random_network():
    """90 atoms placed at random in a 16 A cube with a 2.35 A exclusion.

    A disordered network rather than a crystal, so the ring criteria disagree with each
    other and the counts are sensitive to how each one is defined.
    """
    rng = np.random.default_rng(7)
    box = 16.0
    min_sep = 2.35
    points = []
    while len(points) < 90:
        candidate = rng.random(3) * box
        if points:
            delta = np.array(points) - candidate
            delta -= box * np.round(delta / box)  # minimum image
            if np.min(np.linalg.norm(delta, axis=1)) < min_sep:
                continue
        points.append(candidate)
    return Atoms(f"Si{len(points)}", positions=points, cell=[box] * 3, pbc=True)


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
