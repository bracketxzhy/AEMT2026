"""Micro closed-loop ecosystem environment aligned with the EI paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from config import ExperimentConfig


@dataclass
class StepInfo:
    step: int
    safe: bool
    violation: float
    done_reason: str
    water_recycling_efficiency: float
    oxygen_regeneration_efficiency: float
    load_increment: float
    controls: Dict[str, float]


class MCLSSEEnv:
    """Compact ecological simulator aligned with the EI paper notation."""

    def __init__(
        self,
        config: ExperimentConfig,
        degradation_multiplier: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.system = config.system
        self.reward_cfg = config.reward
        self.degradation_multiplier = float(degradation_multiplier)
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(5, dtype=np.float32)
        self.step_count = 0
        self.done = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self._sample_initial_state()
        self.step_count = 0
        self.done = False
        return self.state.copy()

    def _sample_initial_state(self) -> np.ndarray:
        values = []
        for key in self.system.state_labels:
            low, high = self.system.initial_ranges[key]
            values.append(self._rng.uniform(low, high))
        return np.asarray(values, dtype=np.float32)

    def action_to_control(self, action: np.ndarray) -> Dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 3:
            raise ValueError(f"expected action with 3 entries, received shape {action.shape}")
        action = np.clip(action, 0.0, 1.0)
        return {
            "u_w": float(action[0] * self.system.max_aux_water_support),
            "u_e": float((2.0 * action[1] - 1.0) * self.system.energy_allocation_limit),
            "u_r": float(action[2] * self.system.max_recycling_intensity),
        }

    def _constraint_violation(self, state: np.ndarray) -> float:
        violation = 0.0
        for idx, key in enumerate(("water", "oxygen", "energy", "reliability")):
            bound = self.system.safety_lower_bounds[key]
            violation += max(bound - float(state[idx]), 0.0)
        violation += max(float(state[4]) - self.system.load_limit, 0.0)
        return float(violation)

    def _critical_failure(self, state: np.ndarray) -> bool:
        for idx, key in enumerate(("water", "oxygen", "energy", "reliability")):
            if float(state[idx]) <= self.system.critical_lower_bounds[key]:
                return True
        return bool(float(state[4]) >= self.system.critical_load)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("reset() must be called before stepping a terminated environment")

        control = self.action_to_control(action)
        water, oxygen, energy, reliability, load = [float(x) for x in self.state]
        population_scale = self.system.population_size / 4.0

        energy_support = float(np.clip(0.78 + 0.22 * energy, 0.55, 1.12))
        recycle_activation = 0.58 + 0.42 * control["u_r"]

        water_efficiency = self.system.nominal_water_recycling_efficiency * recycle_activation * reliability * energy_support
        water_efficiency = float(np.clip(water_efficiency, 0.02, 0.98))

        oxygen_efficiency = self.system.nominal_oxygen_regeneration_efficiency * (0.76 + 0.34 * control["u_r"])
        oxygen_efficiency *= reliability * energy_support
        oxygen_efficiency += 0.55 * max(control["u_e"], 0.0)
        oxygen_efficiency = float(np.clip(oxygen_efficiency, 0.02, 1.10))

        water_consumption = self.system.water_consumption_rate * population_scale
        oxygen_consumption = self.system.oxygen_consumption_rate * population_scale

        water_noise = float(self._rng.normal(0.0, self.system.disturbance_scale * 0.22))
        oxygen_noise = float(self._rng.normal(0.0, self.system.disturbance_scale * 0.18))
        energy_noise = float(self._rng.normal(0.0, self.system.disturbance_scale * 0.30))
        reliability_noise = float(self._rng.normal(0.0, self.system.reliability_noise_scale))

        next_water = water - water_consumption + water_consumption * water_efficiency + control["u_w"] + water_noise
        next_oxygen = oxygen - oxygen_consumption + oxygen_consumption * oxygen_efficiency + oxygen_noise

        regulation_energy = self.system.base_energy_demand
        regulation_energy += self.system.water_support_energy_cost * control["u_w"]
        regulation_energy += self.system.recycling_energy_cost * (control["u_r"] ** 1.25)
        regulation_energy += self.system.reliability_energy_drain * max(0.0, 1.0 - reliability)
        next_energy = energy + self.system.base_energy_input + control["u_e"] - regulation_energy + energy_noise

        wear_term = 1.0
        wear_term += self.system.degradation_action_scale * control["u_r"]
        wear_term += 0.45 * (control["u_w"] / self.system.max_aux_water_support)
        wear_term += 0.35 * (abs(control["u_e"]) / self.system.energy_allocation_limit)
        next_reliability = reliability - self.degradation_multiplier * self.system.degradation_rate * wear_term + reliability_noise

        next_water = float(np.clip(next_water, 0.0, self.system.state_upper_clip))
        next_oxygen = float(np.clip(next_oxygen, 0.0, self.system.state_upper_clip))
        next_energy = float(np.clip(next_energy, 0.0, self.system.state_upper_clip))
        next_reliability = float(np.clip(next_reliability, 0.0, 1.05))

        load_increment = 0.0
        load_increment += self.system.load_weights[0] * abs(next_water - 1.0)
        load_increment += self.system.load_weights[1] * abs(next_oxygen - 1.0)
        load_increment += self.system.load_weights[2] * abs(next_energy - 1.0)
        next_load = float(np.clip(self.system.load_retention * load + load_increment, 0.0, self.system.critical_load + 0.6))

        next_state = np.asarray(
            [next_water, next_oxygen, next_energy, next_reliability, next_load],
            dtype=np.float32,
        )

        deviation = abs(next_water - 1.0) + abs(next_oxygen - 1.0) + abs(next_energy - 1.0)
        action_effort = 0.75 * (control["u_w"] / self.system.max_aux_water_support)
        action_effort += 0.50 * (abs(control["u_e"]) / self.system.energy_allocation_limit)
        action_effort += 0.80 * control["u_r"]

        violation = self._constraint_violation(next_state)
        critical_failure = self._critical_failure(next_state)

        buffer_penalty = 0.0
        for idx, key in enumerate(("water", "oxygen", "energy")):
            margin = self.system.safety_lower_bounds[key] + self.reward_cfg.safety_buffer_margin
            buffer_penalty += max(margin - float(next_state[idx]), 0.0)

        reward = self.reward_cfg.survival_bonus
        reward -= self.reward_cfg.ecological_load_penalty * next_load
        reward -= self.reward_cfg.deviation_penalty * deviation
        reward -= self.reward_cfg.reliability_penalty * max(0.0, 1.0 - next_reliability)
        reward -= self.reward_cfg.action_penalty * action_effort
        reward -= self.reward_cfg.safety_buffer_penalty * buffer_penalty
        reward -= self.reward_cfg.safety_violation_penalty * violation

        self.step_count += 1
        reached_horizon = self.step_count >= self.system.episode_length
        done = violation > 0.0 or critical_failure or reached_horizon
        if violation > 0.0 or critical_failure:
            reward -= self.reward_cfg.terminal_penalty

        if critical_failure:
            done_reason = "critical_failure"
        elif violation > 0.0:
            done_reason = "constraint_violation"
        elif reached_horizon:
            done_reason = "horizon"
        else:
            done_reason = "running"

        self.state = next_state
        self.done = done

        info = StepInfo(
            step=self.step_count,
            safe=violation <= 0.0 and not critical_failure,
            violation=violation,
            done_reason=done_reason,
            water_recycling_efficiency=water_efficiency,
            oxygen_regeneration_efficiency=oxygen_efficiency,
            load_increment=load_increment,
            controls=control,
        )
        return next_state.copy(), float(reward), bool(done), info.__dict__



