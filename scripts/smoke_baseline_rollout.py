#!/usr/bin/env python3
"""Baseline-PID smoke rollout for the artificial pancreas simulator.

Runs a single 24-hour episode with zero-delta PID (no RL actor needed)
against the ``controlled`` synthetic profile and prints a small set of
summary metrics. Exit code is ``0`` on success, non-zero on failure.

Intended use:

    python scripts/smoke_baseline_rollout.py

CI / verification (see Phase 9 of the refactor plan) should rely on
this script to detect regressions in the simulator + insulin math
without needing TensorFlow weights to be available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Allow ``import ap_rl`` from a fresh clone without editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    from ap_rl.envs import DiabetesPIDEnv
    from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
    from ap_rl.envs.profile_loader import load_profile
    from ap_rl.envs.scenario_builder import build_scenario
    from ap_rl.runtime.rollout import run_episode
    from ap_rl.utils.paths import configs_dir
    from ap_rl.utils.seed import set_global_seed

    seed = 42
    set_global_seed(seed)

    profile = load_profile(
        "controlled", base_patient_params=dict(DEFAULT_PATIENT_PARAMS)
    )
    meals, exercise = build_scenario(configs_dir() / "meals" / "three_meals_active.yaml")

    env = DiabetesPIDEnv(
        patient_params=profile.patient_params,
        patient_weight=profile.patient_weight,
        target_glucose=profile.target_glucose,
        seed=seed,
        observation_noise_std=profile.observation_noise_std,
        carb_ratio_override=profile.carb_ratio,
        isf_override=profile.isf,
    )
    env.set_meal_schedule(meals)
    env.set_exercise_schedule(exercise)
    env._skip_reload = True
    env.patient_weight = profile.patient_weight

    record = run_episode(env, controller="baseline", max_steps=240, seed=seed)

    stats = record.stats
    print("=== smoke_baseline_rollout ===")
    print(f"profile             : {profile.name}")
    print(f"steps simulated     : {len(record.glucose)}")
    print(f"mean BGL            : {stats.get('mean_glucose', float('nan')):.1f} mg/dL")
    print(f"TIR 70-180          : {stats.get('time_in_range_70_180', float('nan')):.1f}%")
    print(f"time hypo (<70)     : {stats.get('time_hypo_70', float('nan')):.2f}%")
    print(f"total insulin (U)   : {stats.get('total_insulin', float('nan')):.1f}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
