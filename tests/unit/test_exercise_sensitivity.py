from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import yaml

from ap_rl.core.types import PatientConfig
from ap_rl.simulation.hovorka import DEFAULT_HOVORKA_PARAMS, hovorka_ode, pack_params
from ap_rl.simulation.patient import HovorkaPatientModel


BODY_WEIGHT_KG = 75.0
F_PEAK = 1.35
K_RISE_PER_MIN = 5.0 / 60.0
K_DECAY_PER_MIN = 0.01
STATE_NAMES = ("S1", "S2", "I", "x1", "x2", "x3", "Q1", "Q2", "D1", "D2")


def _exercise_patient(f_peak: float = F_PEAK) -> HovorkaPatientModel:
    return HovorkaPatientModel(
        PatientConfig(
            name="exercise-check",
            body_weight_kg=BODY_WEIGHT_KG,
            params={
                "F_peak": f_peak,
                "K_rise": K_RISE_PER_MIN,
                "K_decay": K_DECAY_PER_MIN,
            },
        )
    )


def test_exercise_sensitivity_is_continuous_at_edges_and_decays_to_baseline() -> None:
    patient = _exercise_patient()

    patient._time_min = 0
    patient._update_exercise_sensitivity(True)
    f_at_start = patient._f_sens

    patient._time_min = 600
    patient._update_exercise_sensitivity(True)
    f_at_end = patient._f_sens

    patient._update_exercise_sensitivity(False)
    f_after_end = patient._f_sens

    patient._time_min = 2000
    patient._update_exercise_sensitivity(False)
    f_long_after_end = patient._f_sens

    assert f_at_start == pytest.approx(1.0)
    assert f_at_end == pytest.approx(F_PEAK)
    assert f_after_end == pytest.approx(F_PEAK)
    assert f_long_after_end == pytest.approx(1.0, abs=1e-6)


def test_short_exercise_starts_decay_continuously_at_stop_time() -> None:
    patient = _exercise_patient()

    patient._time_min = 0
    patient._update_exercise_sensitivity(True)
    patient._time_min = 1
    patient._update_exercise_sensitivity(True)
    sensitivity_at_minute_one = patient._f_sens

    patient._time_min = 2
    patient._update_exercise_sensitivity(False)
    sensitivity_at_stop = 1.0 + (F_PEAK - 1.0) * (
        1.0 - np.exp(-K_RISE_PER_MIN * 2.0)
    )

    assert sensitivity_at_minute_one < sensitivity_at_stop < F_PEAK
    assert patient._f_sens == pytest.approx(sensitivity_at_stop)

    patient._time_min = 3
    patient._update_exercise_sensitivity(False)
    assert patient._f_sens == pytest.approx(
        1.0 + (sensitivity_at_stop - 1.0) * np.exp(-K_DECAY_PER_MIN)
    )

    patient.reset()
    assert patient._exercise_end_sensitivity == pytest.approx(1.0)


def test_exercise_multiplier_changes_only_the_three_insulin_activation_terms() -> None:
    state = np.array(
        [0.8, 0.7, 0.02, 0.01, 0.02, 0.03, 96.0, 20.0, 4.0, 6.0],
        dtype=np.float64,
    )
    parameters = pack_params({}, BODY_WEIGHT_KG)
    f_sens = F_PEAK

    base_rhs = hovorka_ode(300.0, state, 0.01, 0.0, 1.0, parameters)
    exercise_rhs = hovorka_ode(300.0, state, 0.01, 0.0, f_sens, parameters)

    insulin = state[2]
    for derivative_index, x_index, k_a_index, k_b_index in (
        (3, 3, 0, 3),
        (4, 4, 1, 4),
        (5, 5, 2, 5),
    ):
        assert exercise_rhs[derivative_index] == pytest.approx(
            f_sens * parameters[k_b_index] * insulin
            - parameters[k_a_index] * state[x_index]
        )

    np.testing.assert_allclose(
        exercise_rhs[[0, 1, 2, 6, 7, 8, 9]],
        base_rhs[[0, 1, 2, 6, 7, 8, 9]],
    )


def test_f_peak_one_preserves_the_base_patient_trajectory() -> None:
    base_patient = _exercise_patient(f_peak=1.0)
    exercised_patient = _exercise_patient(f_peak=1.0)

    for minute in range(180):
        base_state = base_patient.step(
            minute, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=False
        )
        exercised_state = exercised_patient.step(
            minute,
            insulin_rate_u_h=1.0,
            meal_carbs_g=0.0,
            exercise_active=30 <= minute < 90,
        )

    np.testing.assert_allclose(
        [base_state.compartments[name] for name in STATE_NAMES],
        [exercised_state.compartments[name] for name in STATE_NAMES],
    )


def test_exercise_sensitivity_evolves_within_the_first_active_minute() -> None:
    patient = _exercise_patient()
    initial_state = np.array(
        [patient.state.compartments[name] for name in STATE_NAMES], dtype=np.float64
    )
    parameters = pack_params(patient.config.params, BODY_WEIGHT_KG)

    actual = patient.step(
        0, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=True
    )
    reference = solve_ivp(
        lambda time, state: hovorka_ode(
            time,
            state,
            1.0 / 60.0,
            0.0,
            1.0 + (F_PEAK - 1.0) * (1.0 - np.exp(-K_RISE_PER_MIN * time)),
            parameters,
        ),
        (0.0, 1.0),
        initial_state,
        rtol=1e-11,
        atol=1e-13,
    )

    assert reference.success, reference.message
    np.testing.assert_allclose(
        [actual.compartments[name] for name in STATE_NAMES],
        reference.y[:, -1],
        rtol=1e-9,
        atol=1e-11,
    )


def test_empty_config_disables_exercise_extension() -> None:
    assert DEFAULT_HOVORKA_PARAMS["F_peak"] == 1.0
    assert DEFAULT_HOVORKA_PARAMS["K_rise"] == pytest.approx(5.0 / 60.0)
    assert DEFAULT_HOVORKA_PARAMS["K_decay"] == pytest.approx(0.01)

    config = PatientConfig(name="canonical-defaults", params={}, body_weight_kg=75.0)
    base_patient = HovorkaPatientModel(config)
    exercised_patient = HovorkaPatientModel(config)

    for minute in range(3):
        base_state = base_patient.step(
            minute, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=False
        )
        exercised_state = exercised_patient.step(
            minute, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=True
        )

    np.testing.assert_allclose(
        [base_state.compartments[name] for name in STATE_NAMES],
        [exercised_state.compartments[name] for name in STATE_NAMES],
    )


def test_project_exercise_profile_records_the_rate_calibration() -> None:
    profile_path = (
        Path(__file__).parents[2] / "configs" / "profiles" / "project_exercise.yaml"
    )
    profile_text = profile_path.read_text()
    profile = yaml.safe_load(profile_text)

    assert "K_rise = 5 h^-1 = 5 / 60 min^-1" in profile_text
    assert profile == {
        "patient_params": {
            "F_peak": F_PEAK,
            "K_rise": pytest.approx(K_RISE_PER_MIN),
            "K_decay": K_DECAY_PER_MIN,
        }
    }
