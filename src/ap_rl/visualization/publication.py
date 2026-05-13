"""Publication-quality matplotlib plot helpers.

All functions accept an existing ``Axes`` (preferred for compositing) and
also support a one-shot ``save_path`` for quick exports. Sensible
``rcParams`` are applied lazily on first import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg") if not hasattr(matplotlib, "_test_imported") else None  # noqa: E501

import matplotlib.pyplot as plt  # noqa: E402  (after backend resolution)


_PUB_RCPARAMS = {
    "figure.figsize": (8, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
}


def apply_publication_style() -> None:
    """Apply the package-wide publication rcParams (idempotent)."""
    plt.rcParams.update(_PUB_RCPARAMS)


apply_publication_style()


def _maybe_save(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)


def _new_ax(ax: Optional[plt.Axes]) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax


def plot_glucose_trajectory(
    times: Sequence[float],
    glucose: Sequence[float],
    *,
    target: float = 120.0,
    target_band: tuple[float, float] = (70.0, 180.0),
    meals: Optional[Iterable[dict]] = None,
    exercise: Optional[Iterable[dict]] = None,
    ax: Optional[plt.Axes] = None,
    label: str = "Glucose",
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Plot a BGL trajectory with target band, meal markers, exercise spans."""
    fig, ax = _new_ax(ax)
    times = np.asarray(times)
    glucose = np.asarray(glucose)

    ax.axhspan(target_band[0], target_band[1], color="#86d18a", alpha=0.15, label="Target band")
    ax.axhline(target, color="#2a7f3a", linestyle="--", linewidth=1.0, label=f"Target {target:.0f}")
    ax.plot(times, glucose, color="#1f4ea1", label=label)

    if meals:
        meal_list = list(meals)
        for meal in meal_list:
            ax.axvline(meal["time"], color="#e07b00", alpha=0.5, linewidth=1.0)
        if meal_list:
            ax.scatter(
                [m["time"] for m in meal_list],
                [target_band[1] + 10.0] * len(meal_list),
                color="#e07b00",
                marker="^",
                zorder=5,
                label="Meal",
            )
    if exercise:
        active_spans: list[tuple[float, float]] = []
        start: Optional[float] = None
        for ev in exercise:
            if ev["active"] == 1 and start is None:
                start = ev["time"]
            elif ev["active"] == 0 and start is not None:
                active_spans.append((start, ev["time"]))
                start = None
        for span in active_spans:
            ax.axvspan(span[0], span[1], color="#9b59b6", alpha=0.10)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Blood-glucose trajectory")
    ax.legend(loc="upper right")
    _maybe_save(fig, save_path)
    return ax


def plot_insulin(
    times: Sequence[float],
    insulin: Sequence[float],
    *,
    basal: Optional[Sequence[float]] = None,
    bolus: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Plot insulin delivery (total, with optional basal/bolus split)."""
    fig, ax = _new_ax(ax)
    times = np.asarray(times)
    ax.plot(times, insulin, color="#c0392b", label="Total insulin (U/h)")
    if basal is not None:
        ax.plot(times, basal, color="#7f8c8d", linestyle="--", label="Basal")
    if bolus is not None:
        ax.plot(times, bolus, color="#e67e22", linestyle=":", label="Bolus")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Insulin (U/h)")
    ax.set_title("Insulin delivery")
    ax.legend(loc="upper right")
    _maybe_save(fig, save_path)
    return ax


def plot_tir_bar(
    stats: dict,
    *,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Bar chart of time-in-range / hypo / hyper percentages."""
    fig, ax = _new_ax(ax)
    labels = ["Hypo <70", "TIR 70-180", "Hyper >180"]
    values = [
        stats.get("time_hypo_70", 0.0),
        stats.get("time_in_range_70_180", 0.0),
        stats.get("time_hyper_180", 0.0),
    ]
    colors = ["#c0392b", "#27ae60", "#e67e22"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Time (%)")
    ax.set_title("Glucose distribution")
    ax.set_ylim(0, 100)
    for x, v in zip(labels, values):
        ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    _maybe_save(fig, save_path)
    return ax


def plot_reward_curve(
    rewards: Sequence[float],
    *,
    window: int = 20,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Plot per-episode reward with a smoothed rolling mean overlay."""
    fig, ax = _new_ax(ax)
    rewards = np.asarray(rewards, dtype=float)
    ax.plot(rewards, color="#7f8c8d", alpha=0.4, label="Episode reward")
    if rewards.size >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            np.arange(len(smoothed)) + window - 1,
            smoothed,
            color="#1f4ea1",
            label=f"Rolling mean (w={window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Training reward curve")
    ax.legend(loc="lower right")
    _maybe_save(fig, save_path)
    return ax


def plot_controller_comparison(
    times: Sequence[float],
    glucose_by_controller: dict[str, Sequence[float]],
    *,
    target: float = 120.0,
    target_band: tuple[float, float] = (70.0, 180.0),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """Overlay BGL trajectories from multiple controllers."""
    fig, ax = _new_ax(ax)
    ax.axhspan(target_band[0], target_band[1], color="#86d18a", alpha=0.15, label="Target band")
    ax.axhline(target, color="#2a7f3a", linestyle="--", linewidth=1.0)

    palette = ["#1f4ea1", "#c0392b", "#7f8c8d", "#27ae60", "#e67e22"]
    for i, (name, glucose) in enumerate(glucose_by_controller.items()):
        ax.plot(times, glucose, color=palette[i % len(palette)], label=name)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Controller comparison")
    ax.legend(loc="upper right")
    _maybe_save(fig, save_path)
    return ax
