"""Run the full EI-paper experiment pipeline."""

from __future__ import annotations

import torch

from config import get_experiment_config
from controllers.threshold_controller import ThresholdController
from evaluators.evaluate_policy import build_comparison_trajectories, evaluate_controllers
from evaluators.robustness_analysis import run_robustness_analysis
from plots.plot_robustness import plot_robustness
from plots.plot_state_trajectories import plot_ecological_load, plot_state_trajectories
from plots.plot_training_curve import plot_training_curve
from trainers.train_ppo import train_ppo
from utils.common import ensure_directories


def main() -> None:
    config = get_experiment_config()
    ensure_directories([config.output_root, config.figure_dir, config.metric_dir, config.model_dir])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_agent, history = train_ppo(config=config, device=device)
    baseline = ThresholdController(config)

    evaluation_summary, _ = evaluate_controllers(ppo_agent, baseline, config, device=device)
    trajectories = build_comparison_trajectories(ppo_agent, baseline, config, device=device)
    robustness_rows = run_robustness_analysis(ppo_agent, baseline, config, device=device)

    figure_1 = plot_training_curve(history, config.figure_dir / "figure1_training_reward_convergence.png")
    figure_2 = plot_state_trajectories(trajectories, config, config.figure_dir / "figure2_resource_state_trajectories.png")
    figure_3 = plot_ecological_load(trajectories, config.figure_dir / "figure3_ecological_load_evolution.png")
    figure_4 = plot_robustness(robustness_rows, config.figure_dir / "figure4_degradation_robustness.png")

    print("Completed full experiment pipeline.")
    print(f"Figure 1: {figure_1}")
    print(f"Figure 2: {figure_2}")
    print(f"Figure 3: {figure_3}")
    print(f"Figure 4: {figure_4}")
    print(f"Evaluation summary: {config.evaluation_summary_path}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Summary metrics: {evaluation_summary}")


if __name__ == "__main__":
    main()
