"""Evaluation helpers for PPO and threshold-based ecological controllers."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from config import ExperimentConfig
from envs.mclsse_env import MCLSSEEnv
from utils.common import write_csv, write_json


def _controller_action(controller: Any, state: np.ndarray, device: torch.device) -> np.ndarray:
    if hasattr(controller, "act_deterministic"):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = controller.act_deterministic(state_tensor).squeeze(0).cpu().numpy()
        return action.astype(np.float32)
    if hasattr(controller, "act"):
        return np.asarray(controller.act(state), dtype=np.float32)
    raise TypeError(f"Unsupported controller type: {type(controller)!r}")


def simulate_controller(
    controller_name: str,
    controller: Any,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
    degradation_multiplier: float = 1.0,
    record_trajectory: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    env = MCLSSEEnv(config=config, degradation_multiplier=degradation_multiplier, seed=seed)
    state = env.reset(seed=seed)

    trajectory: List[Dict[str, float]] = []
    if record_trajectory:
        trajectory.append(
            {
                "controller": controller_name,
                "step": 0.0,
                "water": float(state[0]),
                "oxygen": float(state[1]),
                "energy": float(state[2]),
                "reliability": float(state[3]),
                "load": float(state[4]),
                "reward": 0.0,
                "u_w": 0.0,
                "u_e": 0.0,
                "u_r": 0.0,
            }
        )

    rewards: List[float] = []
    water_series: List[float] = [float(state[0])]
    oxygen_series: List[float] = [float(state[1])]
    energy_series: List[float] = [float(state[2])]
    load_series: List[float] = [float(state[4])]
    done = False
    done_reason = "running"

    while not done:
        action = _controller_action(controller, state, device)
        next_state, reward, done, info = env.step(action)
        rewards.append(float(reward))
        water_series.append(float(next_state[0]))
        oxygen_series.append(float(next_state[1]))
        energy_series.append(float(next_state[2]))
        load_series.append(float(next_state[4]))
        done_reason = str(info["done_reason"])

        if record_trajectory:
            controls = info["controls"]
            trajectory.append(
                {
                    "controller": controller_name,
                    "step": float(info["step"]),
                    "water": float(next_state[0]),
                    "oxygen": float(next_state[1]),
                    "energy": float(next_state[2]),
                    "reliability": float(next_state[3]),
                    "load": float(next_state[4]),
                    "reward": float(reward),
                    "u_w": float(controls["u_w"]),
                    "u_e": float(controls["u_e"]),
                    "u_r": float(controls["u_r"]),
                }
            )

        state = next_state

    average_load = float(sum(load_series[1:]) / max(1, len(rewards)))
    metrics = {
        "controller": controller_name,
        "seed": float(seed),
        "degradation_multiplier": float(degradation_multiplier),
        "total_reward": float(sum(rewards)),
        "safe_duration": float(len(rewards)),
        "cumulative_load": float(sum(load_series[1:])),
        "average_load": average_load,
        "final_load": float(load_series[-1]),
        "water_std": float(np.std(water_series)),
        "oxygen_std": float(np.std(oxygen_series)),
        "energy_std": float(np.std(energy_series)),
        "safe_completion": float(done_reason == "horizon"),
        "done_reason": done_reason,
    }
    return metrics, trajectory


def _summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for controller in sorted({row["controller"] for row in rows}):
        subset = [row for row in rows if row["controller"] == controller]
        summary[controller] = {
            "mean_total_reward": mean(row["total_reward"] for row in subset),
            "mean_safe_duration": mean(row["safe_duration"] for row in subset),
            "mean_cumulative_load": mean(row["cumulative_load"] for row in subset),
            "mean_average_load": mean(row["average_load"] for row in subset),
            "mean_final_load": mean(row["final_load"] for row in subset),
            "mean_water_std": mean(row["water_std"] for row in subset),
            "mean_oxygen_std": mean(row["oxygen_std"] for row in subset),
            "mean_energy_std": mean(row["energy_std"] for row in subset),
            "safe_completion_rate": mean(row["safe_completion"] for row in subset),
        }
    if {"ppo", "threshold"}.issubset(summary):
        threshold = summary["threshold"]
        ppo = summary["ppo"]
        summary["improvement"] = {
            "safe_duration_gain_pct": 100.0 * (ppo["mean_safe_duration"] - threshold["mean_safe_duration"]) / max(1.0, threshold["mean_safe_duration"]),
            "average_load_reduction_pct": 100.0 * (threshold["mean_average_load"] - ppo["mean_average_load"]) / max(1e-6, threshold["mean_average_load"]),
        }
    return summary


def evaluate_controllers(
    ppo_agent: Any,
    baseline_controller: Any,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, float]] = []
    controller_map = {
        "ppo": ppo_agent,
        "threshold": baseline_controller,
    }

    for controller_name, controller in controller_map.items():
        for run_idx in range(config.evaluation_runs):
            seed = config.ppo.seed + 50_000 + run_idx
            metrics, _ = simulate_controller(
                controller_name=controller_name,
                controller=controller,
                config=config,
                seed=seed,
                device=device,
            )
            rows.append(metrics)

    summary = _summarize(rows)
    write_csv(config.evaluation_runs_path, rows)
    write_json(config.evaluation_summary_path, summary)
    return summary, rows


def build_comparison_trajectories(
    ppo_agent: Any,
    baseline_controller: Any,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> Dict[str, List[Dict[str, float]]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller_map = {
        "ppo": ppo_agent,
        "threshold": baseline_controller,
    }
    combined_rows: List[Dict[str, float]] = []
    trajectory_map: Dict[str, List[Dict[str, float]]] = {}

    for controller_name, controller in controller_map.items():
        _, trajectory = simulate_controller(
            controller_name=controller_name,
            controller=controller,
            config=config,
            seed=config.trajectory_seed,
            device=device,
            record_trajectory=True,
        )
        trajectory_map[controller_name] = trajectory
        combined_rows.extend(trajectory)

    write_csv(config.trajectory_path, combined_rows)
    return trajectory_map

