"""Plot representative resource trajectories and ecological load comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import ExperimentConfig
from plots.common import COLORS, apply_publication_style, finalize_figure


PANEL_TAGS = {"water": "(a)", "oxygen": "(b)", "energy": "(c)"}


def _to_series(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def _termination_step(rows: List[Dict[str, float]]) -> int:
    return max(len(rows) - 1, 0)


def _plot_window(trajectories: Dict[str, List[Dict[str, float]]], margin: int = 8, hard_cap: int = 120) -> int:
    max_term = max(_termination_step(rows) for rows in trajectories.values())
    return min(max_term + margin, hard_cap)


def plot_state_trajectories(
    trajectories: Dict[str, List[Dict[str, float]]],
    config: ExperimentConfig,
    output_path: Path,
) -> Path:
    apply_publication_style()
    xmax = _plot_window(trajectories)
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.0), sharex=True)
    labels: List[Tuple[str, str, float]] = [
        ("water", "Water storage", config.system.safety_lower_bounds["water"]),
        ("oxygen", "Oxygen concentration", config.system.safety_lower_bounds["oxygen"]),
        ("energy", "Energy reserve", config.system.safety_lower_bounds["energy"]),
    ]

    legend_handles = []
    legend_labels = []

    for axis, (key, ylabel, safety_line) in zip(axes, labels):
        axis.axhspan(0.0, safety_line, color=COLORS["unsafe_fill"], alpha=0.72)
        nominal = axis.axhline(1.0, linestyle="--", color=COLORS["muted"], linewidth=1.0)
        safety = axis.axhline(safety_line, linestyle=":", color=COLORS["warning"], linewidth=1.1)

        for controller_name, rows in trajectories.items():
            values = _to_series(rows, key)
            term = min(_termination_step(rows), xmax)
            color = COLORS[controller_name]
            label = "Adaptive PPO" if controller_name == "ppo" else "Threshold baseline"
            clipped_steps = np.arange(min(len(values), xmax + 1), dtype=np.int32)
            line, = axis.plot(clipped_steps, values[: xmax + 1], linewidth=2.0, color=color)
            axis.scatter([term], [values[term]], color=color, s=22, zorder=5)
            axis.axvline(term, color=color, linestyle="--", linewidth=0.9, alpha=0.45)
            axis.annotate(
                f"end {term}",
                (term, values[term]),
                xytext=(5, -10 if controller_name == "ppo" else 10),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
            if key == "water":
                legend_handles.append(line)
                legend_labels.append(label)

        if key == "water":
            legend_handles.extend([nominal, safety])
            legend_labels.extend(["Nominal level", "Safety bound"])

        axis.text(0.01, 0.95, PANEL_TAGS[key], transform=axis.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.55, 1.25)
        axis.grid(True, axis="both")

    axes[0].legend(legend_handles, legend_labels, frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Decision step")
    axes[-1].set_xlim(0, xmax)
    finalize_figure(fig, output_path)
    return output_path


def plot_ecological_load(
    trajectories: Dict[str, List[Dict[str, float]]],
    output_path: Path,
) -> Path:
    apply_publication_style()
    xmax = _plot_window(trajectories, margin=8, hard_cap=140)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for controller_name, rows in trajectories.items():
        steps = [row["step"] for row in rows[: xmax + 1]]
        values = [row["load"] for row in rows[: xmax + 1]]
        color = COLORS[controller_name]
        name = "Adaptive PPO" if controller_name == "ppo" else "Threshold baseline"
        ax.plot(steps, values, linewidth=2.2, color=color, label=name)
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Ecological load")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="both")
    finalize_figure(fig, output_path)
    return output_path
