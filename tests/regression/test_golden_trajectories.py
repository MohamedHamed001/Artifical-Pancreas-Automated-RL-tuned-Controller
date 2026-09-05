import pytest
import numpy as np
from ap_rl.simulation.patient import HovorkaPatientModel
from ap_rl.core.types import PatientConfig

@pytest.fixture
def test_config():
    params = {
        "BW": 75,
        "G_init": 10.0,
        "k_a1": 0.006, "k_a2": 0.06, "k_a3": 0.05,
        "k_b1": 0.003, "k_b2": 0.06, "k_b3": 0.04,
        "V_I": 0.12, "t_max_I": 55, "k_e": 0.138,
        "F_01": 0.0097, "V_G": 0.16, "k_12": 0.066,
        "EGP_0": 0.0161, "AG": 1.0, "t_max_G": 40,
        "A_EGP": 0.05, "phi_EGP": -60, "G_thresh": 9.0, "k_R": 0.0031
    }
    return PatientConfig(name="golden_patient", params=params, body_weight_kg=75.0)

def test_golden_trajectory_basal(test_config):
    """Verify that a 6-hour basal-only run matches the golden value."""
    patient = HovorkaPatientModel(test_config)
    patient.reset()

    # 6 hours = 360 mins
    for t in range(360):
        patient.step(t, insulin_rate_u_h=1.0, meal_carbs_g=0.0, exercise_active=False)

    final_glucose = patient.state.glucose_mgdl
    # Reference value calculated from stabilized physiological model
    expected_glucose = 73.65
    assert final_glucose == pytest.approx(expected_glucose, abs=1e-1)

def test_golden_trajectory_meal(test_config):
    """Verify that a meal event produces the expected glucose peak."""
    patient = HovorkaPatientModel(test_config)
    patient.reset()

    glucose_trace = []
    for t in range(600):
        meal = 50.0 if t == 60 else 0.0
        state = patient.step(t, insulin_rate_u_h=1.0, meal_carbs_g=meal, exercise_active=False)
        glucose_trace.append(state.glucose_mgdl)

    max_glucose = max(glucose_trace)
    # Expected peak for 50g meal with 1U/h basal
    expected_peak = 217.74
    assert max_glucose == pytest.approx(expected_peak, abs=1e-1)
