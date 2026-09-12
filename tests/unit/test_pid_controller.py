import numpy as np

from ap_rl.controllers.pid_controller import BasalOnlyController, PIDController
from ap_rl.core.types import PatientState
from ap_rl.core.types import PatientConfig
from ap_rl.simulation.simulator import SimulationConfig, SimulationRunner


def patient_state(glucose_mgdl: float, time_min: int = 0) -> PatientState:
    return PatientState(
        time_min=time_min,
        glucose_mgdl=glucose_mgdl,
        glucose_rate_mgdl_min=0.0,
        iob_u=0.0,
        cob_g=0.0,
        exercise_active=False,
        compartments={},
    )


def test_physical_glucose_overrides_normalized_observation():
    controller = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, basal_u_h=1.0)
    observation = np.zeros(19, dtype=np.float32)
    observation[0] = 0.6

    action = controller.get_action(
        observation,
        info={"state": patient_state(240.0), "time": 0},
    )

    np.testing.assert_allclose(action, np.array([2.2], dtype=np.float32), atol=1e-6)


def test_pid_integral_uses_simulation_time_delta():
    controller = PIDController(Kp=0.0, Ki=1.0, Kd=0.0, basal_u_h=1.0)
    state = np.array([0.25], dtype=np.float32)

    controller.get_action(state, info={"state": patient_state(119.0), "time": 0})
    controller.get_action(state, info={"state": patient_state(119.0, 5), "time": 5})

    assert controller.pid.ITerm == 5.0


def test_no_info_uses_legacy_state_value_and_action_contract():
    controller = PIDController(Kp=1.0, Ki=0.0, Kd=0.0, basal_u_h=1.0)

    action = controller.get_action(np.array([240.0], dtype=np.float32))

    np.testing.assert_allclose(action, np.array([2.2], dtype=np.float32), atol=1e-6)
    assert action.shape == (1,)
    assert action.dtype == np.float32


def test_reset_restores_constructor_gains_and_clears_pid_state():
    controller = PIDController(Kp=0.5, Ki=0.1, Kd=0.01)
    controller.get_action(
        np.array([0.3], dtype=np.float32),
        info={"state": patient_state(119.0), "time": 0},
    )
    controller.get_action(
        np.array([0.3], dtype=np.float32),
        info={"state": patient_state(119.0, 5), "time": 5},
    )
    assert controller.pid.ITerm != 0.0
    controller.Kp = 2.0
    controller.Ki = 0.5
    controller.Kd = 0.25

    controller.reset()

    assert (controller.Kp, controller.Ki, controller.Kd) == (0.5, 0.1, 0.01)
    assert (controller.pid.PTerm, controller.pid.ITerm, controller.pid.DTerm) == (0.0, 0.0, 0.0)
    assert controller.pid.last_error == 0.0
    assert controller.pid.current_time == 0.0
    assert controller.pid.last_time == 0.0
    assert controller.pid.SetPoint == 120.0


def test_basal_only_returns_configured_rate_for_any_observation_and_after_reset_update():
    controller = BasalOnlyController(1.25)

    np.testing.assert_array_equal(
        controller.get_action(np.zeros(19, dtype=np.float32)),
        np.array([1.25], dtype=np.float32),
    )
    controller.update(reward=-1.0, done=False)
    np.testing.assert_array_equal(
        controller.get_action(np.ones(19, dtype=np.float32)),
        np.array([1.25], dtype=np.float32),
    )
    controller.reset()

    np.testing.assert_array_equal(
        controller.get_action(
            np.array([240.0, -3.0], dtype=np.float32),
            info={"state": patient_state(240.0), "time": 10},
        ),
        np.array([1.25], dtype=np.float32),
    )


def test_basal_only_rejects_negative_nan_and_infinite_rates():
    for rate in (-0.1, np.nan, np.inf, -np.inf):
        with np.testing.assert_raises(ValueError):
            BasalOnlyController(rate)


def test_simulation_runner_supplies_physical_glucose_and_elapsed_time_to_pid():
    controller = PIDController(Kp=0.1, Ki=0.01, Kd=0.001, target_mgdl=120.0)
    runner = SimulationRunner(
        SimulationConfig(
            patient_config=PatientConfig(
                name="pid-contract",
                params={"G_init": 10.0},
                body_weight_kg=75.0,
            ),
            controller=controller,
            duration_min=10,
            dt_min=5,
            seed=42,
        )
    )

    pre_step_glucose = runner.patient.state.glucose_mgdl
    runner.step()
    assert controller.pid.last_error == 120.0 - pre_step_glucose

    runner.step()
    assert controller.pid.last_time == 5.0
