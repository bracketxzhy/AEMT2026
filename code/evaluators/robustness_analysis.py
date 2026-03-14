"""Robustness analysis under different subsystem degradation rates."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

import torch

from config import ExperimentConfig
from evaluators.evaluate_policy import simulate_controller
from utils.common import write_csv


def run_robustness_analysis(
    ppo_agent: Any,
    baseline_controller: Any,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> List[Dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_rows: List[Dict[str, float]] = []
    controller_map = {
        "ppo": ppo_agent,
        "threshold": baseline_controller,
    }

    for multiplier in config.degradation_scenarios:
        for controller_name, controller in controller_map.items():
            runs: List[Dict[str, float]] = []
            for run_idx in range(config.robustness_runs):
                seed = config.ppo.seed + 80_000 + int(multiplier * 1000) + run_idx
                metrics, _ = simulate_controller(
                    controller_name=controller_name,
                    controller=controller,
                    config=config,
                    seed=seed,
                    device=device,
                    degradation_multiplier=multiplier,
                )
                runs.append(metrics)

            aggregated_rows.append(
                {
                    "controller": controller_name,
                    "degradation_multiplier": float(multiplier),
                    "mean_safe_duration": mean(run["safe_duration"] for run in runs),
                    "mean_cumulative_load": mean(run["cumulative_load"] for run in runs),
                    "mean_average_load": mean(run["average_load"] for run in runs),
                    "mean_final_load": mean(run["final_load"] for run in runs),
                    "safe_completion_rate": mean(run["safe_completion"] for run in runs),
                }
            )

    write_csv(config.robustness_path, aggregated_rows)
    return aggregated_rows
