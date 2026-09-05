#!/usr/bin/env python3
"""Capture baseline rollout data for the artificial pancreas simulator.

Runs a single 24-hour episode with zero-delta PID and saves the results
to a JSON file for future comparison.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

# Allow ``import ap_rl`` from a fresh clone without editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
    # Use a standard scenario for baseline capture
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

    # Run for 24 hours (1440 minutes)
    record = run_episode(env, controller="baseline", max_steps=1440, seed=seed)

    # Prepare data for export
    export_data = {
        "metadata": {
            "profile": profile.name,
            "seed": seed,
            "controller": "baseline",
            "target_glucose": env.target_glucose,
        },
        "times": record.times,
        "glucose": record.glucose,
        "insulin": record.insulin,
        "basal": record.basal,
        "bolus": record.bolus,
        "Kp": record.Kp,
        "Ki": record.Ki,
        "Kd": record.Kd,
        "rewards": record.rewards,
        "meals": record.meals,
        "exercise": record.exercise,
        "stats": record.stats,
    }

    output_path = _REPO_ROOT / "baseline_capture.json"
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, cls=NumpyEncoder)

    print(f"Baseline captured and saved to {output_path}")
    print(f"Mean Glucose: {record.stats.get('mean_glucose'):.2f}")
    print(f"TIR 70-180: {record.stats.get('time_in_range_70_180'):.2f}%")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
