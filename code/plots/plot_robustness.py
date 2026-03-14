"""Plot robustness metrics across degradation scenarios for Figure 4."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from plots.common import COLORS, apply_publication_style, finalize_figure


_PANEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.2}


def plot_robustness(rows: List[Dict[str, float]], output_path: Path) -> Path:
    apply_publication_style()
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)

    for values in grouped.values():
        values.sort(key=lambda item: item["degradation_multiplier"])

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.8), sharex=True)
    for controller_name, controller_rows in grouped.items():
        multipliers = [row["degradation_multiplier"] for row in controller_rows]
        safe_duration = [row["mean_safe_duration"] for row in controller_rows]
        cumulative_load = [row["mean_cumulative_load"] for row in controller_rows]
        label = "Adaptive PPO" if controller_name == "ppo" else "Threshold baseline"
        color = COLORS[controller_name]

        axes[0].plot(multipliers, safe_duration, marker="o", markersize=5.5, linewidth=2.1, color=color, label=label)
        axes[1].plot(multipliers, cumulative_load, marker="o", markersize=5.5, linewidth=2.1, color=color, label=label)

    for axis in axes:
        axis.margins(x=0.05, y=0.08)

    axes[0].text(-0.06, 1.04, "(a)", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold", bbox=_PANEL_BOX, clip_on=False)
    axes[0].set_ylabel("Mean safe duration (steps)")
    axes[0].grid(True, axis="both")
    axes[0].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))

    axes[1].text(-0.06, 1.04, "(b)", transform=axes[1].transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold", bbox=_PANEL_BOX, clip_on=False)
    axes[1].set_xlabel("Degradation multiplier")
    axes[1].set_ylabel("Mean cumulative load")
    axes[1].grid(True, axis="both")

    finalize_figure(fig, output_path)
    return output_path
