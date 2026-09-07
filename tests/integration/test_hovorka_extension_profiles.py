"""Deterministic model-layer comparisons, not clinical validation."""

import numpy as np
from scipy.integrate import solve_ivp

from ap_rl.core.types import PatientConfig
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.envs.profile_loader import load_profile
from ap_rl.simulation.hovorka import hovorka_ode, pack_params
from ap_rl.simulation.patient import HovorkaPatientModel


STATE_NAMES = ("S1", "S2", "I", "x1", "x2", "x3", "Q1", "Q2", "D1", "D2")
EXERCISE_START = 30
EXERCISE_STOP = 90
TRACE_MINUTES = 180


def _patient(profile_name: str | None = None, **overrides: float) -> HovorkaPatientModel:
    params = dict(DEFAULT_PATIENT_PARAMS)
    weight = 75.0
    if profile_name is not None:
        profile = load_profile(profile_name, base_patient_params=params)
        params = profile.patient_params
        weight = profile.patient_weight
    params.update(overrides)
    return HovorkaPatientModel(
        PatientConfig(
            name=profile_name or "canonical-base",
            params=params,
            body_weight_kg=weight,
        )
    )


def _trace(patient: HovorkaPatientModel, *, exercise: bool = False) -> np.ndarray:
    states = []
    for minute in range(TRACE_MINUTES):
        state = patient.step(
            minute,
            insulin_rate_u_h=1.0,
            meal_carbs_g=0.0,
            exercise_active=exercise and EXERCISE_START <= minute < EXERCISE_STOP,
        )
        states.append([state.compartments[name] for name in STATE_NAMES])
    return np.asarray(states)


def _reference_exercise_sensitivity(time_min: float) -> float:
    f_peak = 1.35
    k_rise = 5.0 / 60.0
    k_decay = 0.01
    if time_min < EXERCISE_START:
        return 1.0
    if time_min < EXERCISE_STOP:
        return 1.0 + (f_peak - 1.0) * (
            1.0 - np.exp(-k_rise * (time_min - EXERCISE_START))
        )

    sensitivity_at_stop = 1.0 + (f_peak - 1.0) * (
        1.0 - np.exp(-k_rise * (EXERCISE_STOP - EXERCISE_START))
    )
    return 1.0 + (sensitivity_at_stop - 1.0) * np.exp(
        -k_decay * (time_min - EXERCISE_STOP)
    )


def test_zero_amplitude_circadian_profile_matches_canonical_base_trace() -> None:
    base_trace = _trace(_patient())
    circadian_trace = _trace(
        _patient("project_circadian", A_EGP=0.0, phi_EGP=360.0)
    )

    np.testing.assert_array_equal(circadian_trace, base_trace)


def test_unit_peak_exercise_profile_matches_canonical_base_trace() -> None:
    base_trace = _trace(_patient(), exercise=True)
    exercise_trace = _trace(_patient("project_exercise", F_peak=1.0), exercise=True)

    np.testing.assert_array_equal(exercise_trace, base_trace)


def test_exercise_profile_raises_all_insulin_action_states_without_invalid_masses() -> None:
    disabled_trace = _trace(_patient("project_exercise", F_peak=1.0), exercise=True)
    exercise_trace = _trace(_patient("project_exercise"), exercise=True)

    active_endpoint = EXERCISE_STOP - 1
    assert np.all(
        exercise_trace[active_endpoint, 3:6]
        > disabled_trace[active_endpoint, 3:6]
    )
    assert np.all(np.isfinite(exercise_trace))
    assert np.all(exercise_trace >= 0.0)


def test_exercise_profile_minute_steps_match_rk45_through_rise_stop_and_decay() -> None:
    patient = _patient("project_exercise")
    reference_state = np.array(
        [patient.state.compartments[name] for name in STATE_NAMES]
    )
    packed_params = pack_params(patient.config.params, patient.config.body_weight_kg)
    production_states = []
    reference_states = []
    sensitivities = []

    for minute in range(TRACE_MINUTES):
        exercise_active = EXERCISE_START <= minute < EXERCISE_STOP
        sensitivities.append(_reference_exercise_sensitivity(minute))
        production = patient.step(
            minute,
            insulin_rate_u_h=1.0,
            meal_carbs_g=0.0,
            exercise_active=exercise_active,
        )
        reference = solve_ivp(
            lambda time, state: hovorka_ode(
                time,
                state,
                1.0 / 60.0,
                0.0,
                _reference_exercise_sensitivity(time),
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
        production_states.append(
            [production.compartments[name] for name in STATE_NAMES]
        )
        reference_states.append(reference_state.copy())

    production_states = np.asarray(production_states)
    reference_states = np.asarray(reference_states)
    assert sensitivities[EXERCISE_START] == 1.0
    assert sensitivities[EXERCISE_STOP] > sensitivities[EXERCISE_STOP - 1]
    assert sensitivities[EXERCISE_STOP + 1] < sensitivities[EXERCISE_STOP]
    np.testing.assert_allclose(
        production_states[:, 3:6], reference_states[:, 3:6], rtol=0.0, atol=1e-6
    )

    glucose_scale = 18.0182 / (
        patient.config.params["V_G"] * patient.config.body_weight_kg
    )
    glucose_error_mg_dl = (
        production_states[:, 6] - reference_states[:, 6]
    ) * glucose_scale
    assert np.sqrt(np.mean(glucose_error_mg_dl**2)) < 1e-3
    assert np.max(np.abs(glucose_error_mg_dl)) < 1e-3


def test_legacy_profile_trace_is_repeatable_and_distinct_from_the_canonical_base() -> None:
    first_legacy_trace = _trace(_patient("legacy_pre_conformance"))
    second_legacy_trace = _trace(_patient("legacy_pre_conformance"))
    base_trace = _trace(_patient())

    np.testing.assert_array_equal(second_legacy_trace, first_legacy_trace)
    assert not np.array_equal(first_legacy_trace, base_trace)
