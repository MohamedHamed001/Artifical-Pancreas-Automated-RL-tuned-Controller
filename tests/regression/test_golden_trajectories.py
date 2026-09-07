"""Deterministic canonical-base regressions, not clinical validation."""

import pytest

from ap_rl.core.types import PatientConfig
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.simulation.patient import HovorkaPatientModel


@pytest.fixture
def test_config():
    return PatientConfig(
        name="golden_patient",
        params=dict(DEFAULT_PATIENT_PARAMS),
        body_weight_kg=75.0,
    )


def test_golden_trajectory_basal(test_config):
    """Verify that a 6-hour basal-only run matches the golden value."""
    patient = HovorkaPatientModel(test_config)
    patient.reset()

    # 6 hours = 360 mins
    for t in range(360):
        patient.step(t, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=False)

    final_glucose = patient.state.glucose_mgdl
    # Deterministic regression value for the extension-disabled canonical base.
    expected_glucose = 25.78
    assert final_glucose == pytest.approx(expected_glucose, abs=1e-1)


def test_golden_trajectory_meal(test_config):
    """Verify that a meal event produces the expected glucose peak."""
    patient = HovorkaPatientModel(test_config)
    patient.reset()

    meal_minute = 60
    glucose_trace = []
    for t in range(600):
        meal = 50.0 if t == meal_minute else 0.0
        state = patient.step(t, insulin_rate_u_h=1.0, meal_carbs_g=meal, exercise_active=False)
        glucose_trace.append(state.glucose_mgdl)

    post_meal_peak = max(glucose_trace[meal_minute:])
    # Deterministic regression value for the extension-disabled canonical base.
    expected_peak = 146.85
    assert post_meal_peak == pytest.approx(expected_peak, abs=1e-1)
