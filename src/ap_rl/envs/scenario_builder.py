"""Deterministic scenario builder for the Streamlit demo and tests.

Reads a tiny YAML schema and produces in-memory meal / exercise lists
ready for :class:`ap_rl.envs.DiabetesPIDEnv`.

Example schema (``configs/meals/three_meals_active.yaml``)::

    meals:
      - time: 420   # 7:00 - breakfast
        carbs: 45
      - time: 720   # 12:00 - lunch
        carbs: 60
      - time: 1080  # 18:00 - dinner
        carbs: 70
    exercise:
      - start: 1020
        duration: 30
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ap_rl.utils.config import load_yaml


def build_scenario(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load a scenario YAML file into ``(meal_data, exercise_data)`` lists.

    Returned lists are formatted exactly as
    :attr:`DiabetesPIDEnv.meal_data` / ``.exercise_data`` expect.
    """
    cfg = load_yaml(path)
    meals = _coerce_meals(cfg.get("meals", []))
    exercise = _coerce_exercise(cfg.get("exercise", []))
    return meals, exercise


def _coerce_meals(items: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for item in items:
        if "time" not in item or "carbs" not in item:
            raise ValueError(f"meal entry missing 'time' or 'carbs': {item}")
        out.append({"time": float(item["time"]), "carbs": float(item["carbs"])})
    return sorted(out, key=lambda m: m["time"])


def _coerce_exercise(items: Iterable[dict]) -> list[dict]:
    """Convert ``{start, duration}`` rows into start/stop event pairs."""
    out: list[dict] = []
    for item in items:
        if "start" not in item or "duration" not in item:
            raise ValueError(
                f"exercise entry missing 'start' or 'duration': {item}"
            )
        start = float(item["start"])
        duration = float(item["duration"])
        out.append({"time": start, "active": 1})
        out.append({"time": start + duration, "active": 0})
    return sorted(out, key=lambda e: e["time"])
