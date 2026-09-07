"""Numerical integration checks for the current model, not physiology validation."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ap_rl.core.types import PatientConfig
from ap_rl.simulation.hovorka import hovorka_ode, pack_params
from ap_rl.simulation.integrators import rk4_step
from ap_rl.simulation.patient import HovorkaPatientModel


STATE_NAMES = ("S1", "S2", "I", "x1", "x2", "x3", "Q1", "Q2", "D1", "D2")


def _state_vector(patient: HovorkaPatientModel) -> np.ndarray:
    return np.array([patient.state.compartments[name] for name in STATE_NAMES])


def test_rk4_step_matches_coupled_linear_analytic_solution():
    def oscillator(_time: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]])

    actual = rk4_step(oscillator, 0.0, np.array([1.0, 0.0]), 0.1)
    expected = np.array([np.cos(0.1), -np.sin(0.1)])

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_reset_clears_the_previous_episode_glucose_rate():
    config = PatientConfig(name="reset-check", body_weight_kg=75.0, params={})
    patient = HovorkaPatientModel(config)

    stepped_state = patient.step(0, insulin_rate_u_h=1.0, meal_carbs_g=45.0, exercise_active=False)
    assert abs(stepped_state.glucose_rate_mgdl_min) > 1e-6

    reset_state = patient.reset()

    assert reset_state.glucose_rate_mgdl_min == 0.0
    assert reset_state == HovorkaPatientModel(config).state


@pytest.mark.parametrize(
    ("meal_minute", "bolus_start", "circadian_amplitude"),
    [
        pytest.param(None, None, 0.0, id="fasting"),
        pytest.param(480, None, 0.0, id="meal"),
        pytest.param(None, 480, 0.0, id="bolus"),
        pytest.param(480, 480, 0.0, id="meal-and-bolus"),
        pytest.param(None, None, 0.05, id="circadian"),
    ],
)
def test_hovorka_minute_steps_match_rk45(
    meal_minute: int | None,
    bolus_start: int | None,
    circadian_amplitude: float,
):
    config = PatientConfig(
        name="numerical-check",
        body_weight_kg=75.0,
        params={"G_init": 10.0, "A_EGP": circadian_amplitude, "V_G": 0.16},
    )
    patient = HovorkaPatientModel(config)
    reference_state = _state_vector(patient)
    packed_params = pack_params(config.params, config.body_weight_kg)
    production_states = []
    reference_states = []

    for minute in range(24 * 60):
        meal_g = 45.0 if minute == meal_minute else 0.0
        # A 3 U/h square wave for 30 minutes delivers a total 1.5 U bolus.
        bolus_rate_u_h = (
            3.0
            if bolus_start is not None and bolus_start <= minute < bolus_start + 30
            else 0.0
        )
        insulin_rate_u_h = 1.0 + bolus_rate_u_h

        production = patient.step(minute, insulin_rate_u_h, meal_g, exercise_active=False)
        if meal_g:
            reference_state[8] += meal_g * (1000.0 / 180.155)

        reference = solve_ivp(
            lambda time, state: hovorka_ode(
                time,
                state,
                insulin_rate_u_h / 60.0,
                0.0,
                1.0,
                packed_params,
            ),
            (float(minute), float(minute + 1)),
            reference_state,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        assert reference.success, reference.message
        reference_state = reference.y[:, -1]
        production_states.append([production.compartments[name] for name in STATE_NAMES])
        reference_states.append(reference_state.copy())

    production_states = np.asarray(production_states)
    reference_states = np.asarray(reference_states)
    assert np.all(np.isfinite(production_states))
    assert np.all(np.isfinite(reference_states))
    assert np.all(production_states >= 0.0)
    assert np.all(reference_states >= 0.0)

    glucose_scale = 18.0182 / (config.params["V_G"] * config.body_weight_kg)
    glucose_error_mg_dl = (production_states[:, 6] - reference_states[:, 6]) * glucose_scale
    # Numerical regression bounds, not clinical accuracy thresholds.
    assert np.sqrt(np.mean(glucose_error_mg_dl**2)) < 1e-3
    assert np.max(np.abs(glucose_error_mg_dl)) < 1e-3
