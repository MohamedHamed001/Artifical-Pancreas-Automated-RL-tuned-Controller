"""Glucose trajectory metrics shared by the env, demo, and publication plots."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_float_array(glucose: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(glucose, dtype=float)
    if arr.ndim != 1:
        raise ValueError("glucose must be a 1-D sequence")
    return arr


def percent_in_band(glucose: Sequence[float] | np.ndarray, low: float, high: float) -> float:
    """Fraction of samples in [low, high], as a percentage (0–100)."""
    g = _as_float_array(glucose)
    if g.size == 0:
        return 0.0
    return float(np.sum((g >= low) & (g <= high)) / len(g) * 100.0)


def glucose_trajectory_summary(glucose: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Aggregate glucose-only statistics used by :meth:`DiabetesPIDEnv.get_statistics`.

    Keeps band definitions aligned with the legacy reward / reporting
    convention (80–140 tight band, 70–180 clinical TIR).
    """
    g = _as_float_array(glucose)
    if g.size == 0:
        return {}
    return {
        "mean_glucose": float(np.mean(g)),
        "std_glucose": float(np.std(g)),
        "time_in_range_80_140": percent_in_band(g, 80.0, 140.0),
        "time_in_range_70_180": percent_in_band(g, 70.0, 180.0),
        "time_hypo_70": float(np.sum(g < 70.0) / len(g) * 100.0),
        "time_hyper_180": float(np.sum(g > 180.0) / len(g) * 100.0),
    }
