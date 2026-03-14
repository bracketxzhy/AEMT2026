"""Plot PPO training reward convergence for the manuscript training figure."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import matplotlib.pyplot as plt
import numpy as np

from plots.common import COLORS, apply_publication_style, finalize_figure
from utils.common import moving_average


def plot_training_curve(history: List[Mapping[str, float]], output_path: Path) -> Path:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    iterations = np.asarray([row["iteration"] for row in history], dtype=np.float32)
    returns = np.asarray([row["mean_return"] for row in history], dtype=np.float32)
    smoothed = moving_average(returns, window=max(7, len(returns) // 12))

    residual = returns - smoothed
    band = np.sqrt(np.maximum(moving_average(residual ** 2, window=max(7, len(returns) // 12)), 0.0))

    ax.fill_between(iterations, smoothed - band, smoothed + band, color=COLORS["ppo_light"], alpha=0.75, linewidth=0)
    ax.plot(iterations, returns, color=COLORS["ppo_light"], linewidth=1.0, alpha=1.0)
    ax.plot(iterations, smoothed, color=COLORS["ppo"], linewidth=2.4)

    best_idx = int(np.argmax(smoothed))
    ax.scatter(iterations[best_idx], smoothed[best_idx], color=COLORS["accent"], s=30, zorder=5)
    ax.annotate(
        f"best = {smoothed[best_idx]:.2f}",
        (iterations[best_idx], smoothed[best_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["accent"],
    )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean episode return")
    ax.grid(True, axis="both")
    finalize_figure(fig, output_path)
    return output_path
