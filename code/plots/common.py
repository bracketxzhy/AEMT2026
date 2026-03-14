"""Shared matplotlib style for publication-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "ppo": "#1E78B5",
    "ppo_light": "#E0E7F3",
    "threshold": "#CB3535",
    "threshold_light": "#F5E0E0",
    "accent": "#6E7781",
    "muted": "#7A8796",
    "warning": "#CB3535",
    "grid": "#D9E1ED",
    "unsafe_fill": "#F5E0E0",
}


def apply_publication_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "grid.alpha": 0.6,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def finalize_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
