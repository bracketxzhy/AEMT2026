"""Shared helpers for reproducibility and artifact persistence."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


def ensure_directories(paths: Sequence[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            raise ValueError("fieldnames must be provided when rows are empty")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    actual_fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    if window <= 1:
        return values
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    averaged = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, averaged[0], dtype=np.float32)
    return np.concatenate([prefix, averaged])
