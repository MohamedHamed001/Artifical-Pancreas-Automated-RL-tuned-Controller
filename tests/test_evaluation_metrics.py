from __future__ import annotations

import numpy as np
import pytest

from ap_rl.evaluation.metrics import glucose_trajectory_summary, percent_in_band


def test_percent_in_band_empty() -> None:
    assert percent_in_band([], 70, 180) == 0.0


def test_glucose_trajectory_summary_matches_env_formula() -> None:
    g = np.array([60.0, 100.0, 200.0, 120.0])
    s = glucose_trajectory_summary(g)
    assert s["mean_glucose"] == pytest.approx(np.mean(g))
    assert s["time_hypo_70"] == pytest.approx(25.0)
    assert s["time_hyper_180"] == pytest.approx(25.0)
    assert s["time_in_range_70_180"] == pytest.approx(50.0)
    assert s["time_in_range_80_140"] == pytest.approx(50.0)
