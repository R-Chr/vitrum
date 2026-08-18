"""Generate figure1.png for the JOSS paper.

Runs three of the analyses shown in examples/analysis/demo.ipynb on the sodium
silicate trajectory shipped with the package, moving from the diffraction
pattern through short-range order to the medium-range ring statistics. Only the
core dependencies are needed.

Usage:
    python paper/make_figure.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from vitrum import Coordination, RingAnalysis, Scattering
from vitrum.io_helpers import correct_atom_types

REPO = Path(__file__).resolve().parent.parent
TRAJECTORY = REPO / "examples" / "analysis" / "md.lammpstrj"
OUTPUT = Path(__file__).resolve().parent / "figure1.png"

ATOM_TYPES = {1: "Na", 2: "O", 3: "Si"}
RING_CRITERIA = ("guttman", "king")
BLUE, ORANGE, GREEN = "#0173B2", "#DE8F05", "#029E73"

# The ideal tetrahedral angle, for reference in panel (b).
TETRAHEDRAL = 109.47

STYLE = {
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 4.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.35,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#4d4d4d",
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
}


def panel_label(ax, letter):
    """Bold panel letter, set just outside the top-left corner of the axes."""
    ax.text(
        -0.18,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_structure_factor(ax, atoms):
    """(a) Total structure factor under neutron and X-ray weighting."""
    scattering = Scattering(atoms, rrange=15, qmax=20, nbin=500, disable_progress=True)
    ax.axhline(1.0, color="#c8c8c8", lw=0.7, zorder=0)
    for weighting, colour, label in (
        ("neutron", BLUE, "Neutron"),
        ("xray", ORANGE, "X-ray"),
    ):
        ax.plot(
            scattering.qval,
            scattering.get_structure_factor(type=weighting),
            color=colour,
            label=label,
        )
    ax.set_xlabel(r"$Q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$S(Q)$")
    ax.set_xlim(0, 20)
    ax.legend(loc="upper right")


def plot_angle_distributions(ax, atoms):
    """(b) Intra- and inter-tetrahedral bond angle distributions."""
    coordination = Coordination(atoms)
    ax.axvline(TETRAHEDRAL, color="#c8c8c8", lw=0.7, ls=(0, (4, 2)), zorder=0)
    for (center, neighbours), colour in (
        (("Si", ["O", "O"]), BLUE),
        (("O", ["Si", "Si"]), ORANGE),
    ):
        angles, density = coordination.get_angle_distribution(
            center, neighbours, nbin=90, range=(0, 180)
        )
        label = f"{neighbours[0]}–{center}–{neighbours[1]}"
        ax.plot(angles, density, color=colour, label=label)
    ax.set_xlabel("Bond angle (°)")
    ax.set_ylabel("Probability density (°$^{-1}$)")
    ax.set_xlim(60, 180)
    ax.set_xticks([60, 90, 120, 150, 180])
    ax.text(
        TETRAHEDRAL + 3,
        ax.get_ylim()[1] * 0.97,
        f"tetrahedral, {TETRAHEDRAL}°",
        fontsize=7.5,
        color="#767676",
        ha="left",
        va="top",
    )
    ax.legend(loc="upper left")


def plot_ring_distribution(ax, atoms):
    """(c) Ring size distribution of the Si-O network in the final frame."""
    rings = RingAnalysis(atoms[-1], included_atoms=["Si", "O"], bonding_dict=[("Si", "O")])
    for criterion, colour in zip(RING_CRITERIA, (BLUE, GREEN)):
        rings.calculate(radii_factor=1.3, max_size=20, criterion=criterion)
        distribution = rings.get_ring_size_distribution()
        sizes = sorted(distribution)
        frequency = np.array([distribution[n] for n in sizes]) / rings.atoms.get_volume()
        ax.plot(
            sizes,
            1e3 * frequency,
            color=colour,
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=criterion.capitalize(),
        )
    ax.set_xlabel("Ring size ($N_\\mathrm{atoms}$)")
    ax.set_ylabel(r"Ring density ($10^{-3}$ Å$^{-3}$)")
    ax.set_xticks(range(8, 21, 2))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")


def main():
    atoms = read(TRAJECTORY, index=":10", format="lammps-dump-text")
    correct_atom_types(atoms, ATOM_TYPES)

    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

        plot_structure_factor(axes[0], atoms)
        plot_angle_distributions(axes[1], atoms)
        plot_ring_distribution(axes[2], atoms)

        for ax, letter in zip(axes, "abc"):
            panel_label(ax, letter)

        fig.tight_layout(w_pad=2.4)
        fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")

    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
