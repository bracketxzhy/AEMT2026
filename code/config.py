"""Central configuration for the micro closed-loop ecosystem model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SystemConfig:
    """Physical and operational parameters for the ecological simulator."""

    population_size: int = 4
    episode_length: int = 500
    state_labels: Tuple[str, ...] = ("water", "oxygen", "energy", "reliability", "load")
    nominal_state: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 0.0)
    initial_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "water": (0.94, 1.06),
            "oxygen": (0.95, 1.05),
            "energy": (0.92, 1.08),
            "reliability": (0.95, 1.00),
            "load": (0.02, 0.08),
        }
    )
    water_consumption_rate: float = 0.045
    oxygen_consumption_rate: float = 0.040
    nominal_water_recycling_efficiency: float = 0.86
    nominal_oxygen_regeneration_efficiency: float = 1.02
    max_aux_water_support: float = 0.080
    energy_allocation_limit: float = 0.040
    max_recycling_intensity: float = 1.000
    base_energy_input: float = 0.072
    base_energy_demand: float = 0.045
    water_support_energy_cost: float = 0.500
    recycling_energy_cost: float = 0.028
    reliability_energy_drain: float = 0.020
    degradation_rate: float = 0.00058
    degradation_action_scale: float = 0.75
    disturbance_scale: float = 0.006
    reliability_noise_scale: float = 0.00035
    load_retention: float = 0.994
    load_weights: Tuple[float, float, float] = (0.18, 0.22, 0.15)
    safety_lower_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "water": 0.72,
            "oxygen": 0.76,
            "energy": 0.68,
            "reliability": 0.34,
        }
    )
    critical_lower_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "water": 0.62,
            "oxygen": 0.66,
            "energy": 0.55,
            "reliability": 0.20,
        }
    )
    load_limit: float = 1.50
    critical_load: float = 1.90
    state_upper_clip: float = 1.60


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the CMDP-style reward shaping."""

    survival_bonus: float = 1.35
    ecological_load_penalty: float = 1.25
    deviation_penalty: float = 0.95
    action_penalty: float = 0.08
    reliability_penalty: float = 0.25
    safety_violation_penalty: float = 20.0
    safety_buffer_penalty: float = 6.0
    safety_buffer_margin: float = 0.08
    terminal_penalty: float = 25.0


@dataclass(frozen=True)
class PPOConfig:
    """Training hyperparameters for PPO."""

    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    grad_clip: float = 0.80
    batch_size: int = 64
    rollout_episodes_per_iteration: int = 4
    update_epochs: int = 10
    training_iterations: int = 90
    seed: int = 2026


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration and output locations."""

    system: SystemConfig = field(default_factory=SystemConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation_runs: int = 50
    robustness_runs: int = 40
    degradation_scenarios: Tuple[float, ...] = (0.75, 1.00, 1.25, 1.50, 1.75)
    trajectory_seed: int = 37
    output_root: Path = ROOT_DIR / "outputs"
    figure_dir: Path = ROOT_DIR / "outputs" / "figures"
    metric_dir: Path = ROOT_DIR / "outputs" / "metrics"
    model_dir: Path = ROOT_DIR / "outputs" / "models"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["figure_dir"] = str(self.figure_dir)
        data["metric_dir"] = str(self.metric_dir)
        data["model_dir"] = str(self.model_dir)
        data["degradation_scenarios"] = list(self.degradation_scenarios)
        return data

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "ppo_space_station.pt"

    @property
    def training_history_path(self) -> Path:
        return self.metric_dir / "training_history.csv"

    @property
    def evaluation_summary_path(self) -> Path:
        return self.metric_dir / "evaluation_summary.json"

    @property
    def evaluation_runs_path(self) -> Path:
        return self.metric_dir / "evaluation_runs.csv"

    @property
    def trajectory_path(self) -> Path:
        return self.metric_dir / "trajectory_comparison.csv"

    @property
    def robustness_path(self) -> Path:
        return self.metric_dir / "robustness_metrics.csv"

    @property
    def config_snapshot_path(self) -> Path:
        return self.metric_dir / "experiment_config.json"


def get_experiment_config() -> ExperimentConfig:
    """Build the default experiment configuration used across the project."""

    return ExperimentConfig()


def scenario_rows(config: ExperimentConfig) -> List[Dict[str, float]]:
    """Return degradation scenarios in a plotting-friendly structure."""

    return [
        {"degradation_multiplier": multiplier, "effective_rate": multiplier * config.system.degradation_rate}
        for multiplier in config.degradation_scenarios
    ]



