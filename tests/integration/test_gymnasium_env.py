import pytest
import numpy as np
from ap_rl.core.types import PatientConfig
from ap_rl.envs.glucose_control_env import GlucoseControlEnv

def test_gym_env_step():
    """Verify that the Gymnasium environment can be reset and stepped."""
    p_config = PatientConfig(name="test_patient", params={"G_init": 10.0}, body_weight_kg=75.0)
    env = GlucoseControlEnv(patient_config=p_config)

    obs, info = env.reset(seed=42)
    assert obs.shape == (6,)
    assert "glucose" in info

    # Take a step with zero insulin
    action = np.array([0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (6,)
    assert isinstance(reward, (float, np.float32, np.float64))
    # Gymnasium might return np.bool_
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert info["time"] == 5  # dt_min = 5

def test_gym_env_reproducibility():
    """Verify that the Gymnasium environment is deterministic with seeds."""
    p_config = PatientConfig(name="test_patient", params={"G_init": 10.0}, body_weight_kg=75.0)
    env = GlucoseControlEnv(patient_config=p_config)

    obs1, _ = env.reset(seed=123)
    action = np.array([1.0], dtype=np.float32)
    next_obs1, _, _, _, _ = env.step(action)

    obs2, _ = env.reset(seed=123)
    next_obs2, _, _, _, _ = env.step(action)

    np.testing.assert_array_almost_equal(obs1, obs2)
    np.testing.assert_array_almost_equal(next_obs1, next_obs2)

if __name__ == "__main__":
    test_gym_env_step()
    test_gym_env_reproducibility()
    print("Gymnasium env tests passed!")
