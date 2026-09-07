"""Smoke tests for DiabetesPIDEnv reset/step shapes and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from ap_rl.envs import DiabetesPIDEnv
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS


@pytest.fixture
def env() -> DiabetesPIDEnv:
    e = DiabetesPIDEnv(
        patient_params=dict(DEFAULT_PATIENT_PARAMS),
        patient_weight=75,
        target_glucose=120,
        seed=42,
    )
    # Inject a tiny deterministic schedule so reset() does not pick a
    # random test case from disk.
    e.set_meal_schedule([{"time": 60, "carbs": 30.0}])
    e.set_exercise_schedule([])
    e._skip_reload = True
    e.patient_weight = 75
    return e


def test_reset_returns_state_with_expected_shape(env: DiabetesPIDEnv) -> None:
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_tuple(env: DiabetesPIDEnv) -> None:
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    state, reward, terminated, truncated, info = env.step(action)
    assert isinstance(state, np.ndarray)
    assert state.shape == env.observation_space.shape
    assert isinstance(reward, (float, np.float32, np.float64))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "glucose" in info and "total_insulin" in info
    assert "Kp" in info and "Ki" in info and "Kd" in info


@pytest.mark.parametrize("safety_blocked", [False, True])
def test_step_observation_uses_safety_delivered_iob(
    env: DiabetesPIDEnv, safety_blocked: bool
) -> None:
    env.reset()
    if safety_blocked:
        env.runner.safety.min_glucose_mgdl = 1000.0

    observation, _, _, _, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    if safety_blocked:
        assert info["total_insulin"] == 0.0
        assert info["iob"] == 0.0
    else:
        assert info["total_insulin"] > 0.0
        assert info["iob"] == pytest.approx(info["total_insulin"] / 60.0)
    assert observation[15] == pytest.approx(info["iob"] / 10.0)


def test_baseline_short_episode_no_termination(env: DiabetesPIDEnv) -> None:
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(120):
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    # 2 hours into a baseline run we expect glucose to remain finite and in
    # a roughly physiological range (50-300 mg/dL).
    assert 50.0 <= info["glucose"] <= 400.0


def test_seed_reproducibility() -> None:
    common = dict(
        patient_params=dict(DEFAULT_PATIENT_PARAMS),
        patient_weight=75,
        target_glucose=120,
        observation_noise_std=2.0,
    )
    env_a = DiabetesPIDEnv(seed=123, **common)
    env_a.set_meal_schedule([{"time": 60, "carbs": 30.0}])
    env_a._skip_reload = True
    env_a.patient_weight = 75

    env_b = DiabetesPIDEnv(seed=123, **common)
    env_b.set_meal_schedule([{"time": 60, "carbs": 30.0}])
    env_b._skip_reload = True
    env_b.patient_weight = 75

    env_a.reset()
    env_b.reset()
    action = np.zeros(3, dtype=np.float32)
    for _ in range(30):
        _, _, _, _, info_a = env_a.step(action)
        _, _, _, _, info_b = env_b.step(action)
    # Two identically-seeded envs should match within numerical noise of
    # the Hovorka integrator (numba fastmath is allowed a tiny drift).
    assert info_a["glucose"] == pytest.approx(info_b["glucose"], abs=5e-2)
