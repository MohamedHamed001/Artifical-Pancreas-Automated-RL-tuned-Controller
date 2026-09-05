"""Glucose trajectory metrics shared by the env, demo, and publication plots."""

from __future__ import annotations
from typing import Sequence, Dict
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


def risk_indices(glucose: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """
    Computes LBGI, HBGI, and BGRI according to Kovatchev et al.
    Returns (LBGI, HBGI, BGRI).
    """
    g = _as_float_array(glucose)
    if g.size == 0:
        return 0.0, 0.0, 0.0

    # Kovatchev risk function mapping
    # G must be in mg/dL
    # Avoid log(0)
    g_safe = np.maximum(g, 1.0)
    f = 1.509 * (np.power(np.log(g_safe), 1.084) - 5.381)

    risk = 10 * f**2

    lbgi = np.mean(np.where(f < 0, risk, 0))
    hbgi = np.mean(np.where(f > 0, risk, 0))
    bgri = lbgi + hbgi

    return float(lbgi), float(hbgi), float(bgri)


def glucose_trajectory_summary(glucose: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Aggregate glucose statistics including TIR, BGRI, and CV.
    """
    g = _as_float_array(glucose)
    if g.size == 0:
        return {}

    mean_g = np.mean(g)
    std_g = np.std(g)
    cv = (std_g / mean_g * 100.0) if mean_g > 0 else 0.0

    lbgi, hbgi, bgri = risk_indices(g)

    return {
        "mean_glucose": float(mean_g),
        "std_glucose": float(std_g),
        "cv": float(cv),
        "tir_70_180": percent_in_band(g, 70.0, 180.0),
        "tbr_70": float(np.sum(g < 70.0) / len(g) * 100.0),
        "tar_180": float(np.sum(g > 180.0) / len(g) * 100.0),
        "tbr_54": float(np.sum(g < 54.0) / len(g) * 100.0),
        "tar_250": float(np.sum(g > 250.0) / len(g) * 100.0),
        # Legacy key aliases
        "time_in_range_70_180": percent_in_band(g, 70.0, 180.0),
        "time_in_range_80_140": percent_in_band(g, 80.0, 140.0),
        "time_hypo_70": float(np.sum(g < 70.0) / len(g) * 100.0),
        "time_hyper_180": float(np.sum(g > 180.0) / len(g) * 100.0),
        "lbgi": lbgi,
        "hbgi": hbgi,
        "bgri": bgri,
    }
