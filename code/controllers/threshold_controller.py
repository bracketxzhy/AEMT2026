"""Rule-based threshold controller used as the baseline comparator."""

from __future__ import annotations

import numpy as np

from config import ExperimentConfig


class ThresholdController:
    """Reactive controller that intervenes only after thresholds are crossed."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def act(self, state: np.ndarray) -> np.ndarray:
        water, oxygen, energy, reliability, load = [float(x) for x in state]

        aux_water = 0.04
        energy_bias = 0.50
        recycle = 0.30

        if water < 0.90:
            aux_water += 0.24
        if water < 0.80:
            aux_water += 0.30
            recycle += 0.10
        if water < 0.74:
            aux_water += 0.22
            recycle += 0.14

        if oxygen < 0.87:
            recycle += 0.22
            energy_bias += 0.08
        if oxygen < 0.79:
            recycle += 0.26
            energy_bias += 0.12

        if energy < 0.82:
            energy_bias += 0.22
            recycle -= 0.05
        if energy < 0.70:
            energy_bias += 0.24
            recycle -= 0.12
            aux_water -= 0.04

        if reliability < 0.78:
            recycle = min(recycle, 0.60)
            energy_bias += 0.05
        if reliability < 0.62:
            recycle = min(recycle, 0.50)
            energy_bias += 0.08

        if load > 1.05:
            aux_water += 0.06
            recycle += 0.08
        if load > 1.25:
            aux_water += 0.08
            energy_bias += 0.08

        action = np.asarray([
            np.clip(aux_water, 0.0, 1.0),
            np.clip(energy_bias, 0.0, 1.0),
            np.clip(recycle, 0.0, 1.0),
        ], dtype=np.float32)
        return action
