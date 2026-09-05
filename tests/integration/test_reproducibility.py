import pytest
import numpy as np
from ap_rl.core.types import PatientConfig
from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.controllers.pid_controller import PIDController

def test_simulation_determinism():
    """Verify that the same seed and config produce identical trajectories."""
    p_config = PatientConfig(name="test_patient", params={"G_init": 10.0}, body_weight_kg=75.0)
    controller = PIDController(Kp=0.1, Ki=0.01, Kd=0.001)

    config = SimulationConfig(
        patient_config=p_config,
        controller=controller,
        duration_min=120,
        seed=123
    )

    runner1 = SimulationRunner(config)
    res1 = runner1.run()

    runner2 = SimulationRunner(config)
    res2 = runner2.run()

    # Check glucose trajectory
    np.testing.assert_array_almost_equal(
        res1.glucose,
        res2.glucose,
        err_msg="Glucose trajectories diverge despite identical seed."
    )

    # Check insulin trajectory
    np.testing.assert_array_almost_equal(
        res1.insulin,
        res2.insulin,
        err_msg="Insulin trajectories diverge despite identical seed."
    )

if __name__ == "__main__":
    test_simulation_determinism()
    print("Determinism test passed!")
