"""Headless smoke test for the publication plot helpers."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ap_rl.visualization.publication import (  # noqa: E402
    apply_publication_style,
    plot_controller_comparison,
    plot_glucose_trajectory,
    plot_insulin,
    plot_reward_curve,
    plot_tir_bar,
)


@pytest.fixture(scope="module")
def fake_episode() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    times = np.arange(120, dtype=float)
    glucose = 120 + 25 * np.sin(times / 15.0) + rng.normal(0, 1.0, size=times.shape)
    insulin = 0.9 + 0.1 * rng.standard_normal(size=times.shape)
    rewards = rng.standard_normal(60).cumsum()
    return {
        "times": times,
        "glucose": glucose,
        "insulin": insulin,
        "rewards": rewards,
    }


def test_apply_publication_style_idempotent() -> None:
    apply_publication_style()
    apply_publication_style()


def test_plot_glucose_returns_axes(fake_episode) -> None:
    ax = plot_glucose_trajectory(
        fake_episode["times"],
        fake_episode["glucose"],
        meals=[{"time": 30, "carbs": 45}],
        exercise=[{"time": 60, "active": 1}, {"time": 90, "active": 0}],
    )
    assert ax is not None
    assert ax.get_xlabel() == "Time (min)"


def test_plot_insulin_returns_axes(fake_episode) -> None:
    ax = plot_insulin(fake_episode["times"], fake_episode["insulin"])
    assert ax is not None


def test_plot_tir_bar_runs() -> None:
    stats = {"time_hypo_70": 2.5, "time_in_range_70_180": 90.0, "time_hyper_180": 7.5}
    ax = plot_tir_bar(stats)
    assert ax is not None


def test_plot_reward_curve_runs(fake_episode) -> None:
    ax = plot_reward_curve(fake_episode["rewards"], window=10)
    assert ax is not None


def test_plot_controller_comparison_runs(fake_episode) -> None:
    ax = plot_controller_comparison(
        fake_episode["times"],
        {
            "baseline": fake_episode["glucose"],
            "rl": fake_episode["glucose"] - 5,
        },
    )
    assert ax is not None


def test_savefig_creates_file(tmp_path, fake_episode) -> None:
    path = tmp_path / "subdir" / "glucose.png"
    plot_glucose_trajectory(
        fake_episode["times"],
        fake_episode["glucose"],
        save_path=path,
    )
    assert path.exists() and path.stat().st_size > 0
    # Ensure save_path also works for the comparison helper.
    cmp_path = tmp_path / "cmp.png"
    plot_controller_comparison(
        fake_episode["times"],
        {"baseline": fake_episode["glucose"], "rl": fake_episode["glucose"] + 5},
        save_path=cmp_path,
    )
    assert cmp_path.exists()


def test_no_display_required(monkeypatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    assert os.environ.get("MPLBACKEND") == "Agg"
