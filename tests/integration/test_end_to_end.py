import pytest
import numpy as np
from ap_rl.core.types import PatientConfig
from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.controllers.pid_controller import PIDController
from ap_rl.controllers.mpc_controller import GlucoseMPC
from ap_rl.envs.glucose_control_env import GlucoseControlEnv

def test_full_simulation_with_pid():
    """Verify that a full simulation runs with the PID controller."""
    p_config = PatientConfig(name="test_patient", params={"G_init": 10.0}, body_weight_kg=75.0)
    controller = PIDController(Kp=0.1, Ki=0.01, Kd=0.001)

    config = SimulationConfig(
        patient_config=p_config,
        controller=controller,
        duration_min=120,  # 2 hours
        seed=1
    )

    runner = SimulationRunner(config=config)

    record = runner.run()
    assert len(record.steps) > 0
    # Use the property to get glucose array
    assert record.glucose[-1] > 0
    print(f"PID Final Glucose: {record.glucose[-1]}")

def test_gym_env_with_mpc():
    """Verify that the Gymnasium environment works with the MPC controller."""
    p_config = PatientConfig(name="test_patient", params={"G_init": 10.0}, body_weight_kg=75.0)
    env = GlucoseControlEnv(patient_config=p_config)
    controller = GlucoseMPC(horizon=6) # Small horizon for speed

    obs, info = env.reset(seed=42)
    for _ in range(5):
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None
        if terminated or truncated:
            break

    assert info["time"] > 0
    print(f"MPC simulation successful.")

if __name__ == "__main__":
    test_full_simulation_with_pid()
    test_gym_env_with_mpc()
