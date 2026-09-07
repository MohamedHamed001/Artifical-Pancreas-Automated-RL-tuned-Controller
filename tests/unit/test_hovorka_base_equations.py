import numpy as np
import pytest

from ap_rl.core.types import PatientConfig
from ap_rl.simulation.hovorka import hovorka_ode, pack_params
from ap_rl.simulation.patient import HovorkaPatientModel


BODY_WEIGHT_KG = 75.0
V_G_L_PER_KG = 0.16


def _base_parameters(**overrides: float) -> np.ndarray:
    parameters = {
        "k_a1": 0.006,
        "k_a2": 0.06,
        "k_a3": 0.03,
        "k_b1": 0.03072,
        "k_b2": 0.0492,
        "k_b3": 1.56,
        "V_I": 0.12,
        "t_max_I": 55.0,
        "k_e": 0.138,
        "F_01": 0.0097,
        "V_G": V_G_L_PER_KG,
        "k_12": 0.066,
        "EGP_0": 0.0161,
        "AG": 0.8,
        "t_max_G": 40.0,
        "A_EGP": 0.0,
        "phi_EGP": -60.0,
        "G_thresh": 9.0,
        "k_R": 0.0031,
    }
    parameters.update(overrides)
    return pack_params(parameters, BODY_WEIGHT_KG)


def test_glucose_compartments_follow_hovorka_transfer_equations() -> None:
    state = np.array(
        [0.8, 0.7, 0.02, 0.01, 0.02, 0.03, 96.0, 20.0, 4.0, 6.0],
        dtype=np.float64,
    )

    derivative = hovorka_ode(0.0, state, 0.01, 0.0, 1.0, _base_parameters())

    u_id = 0.8 * 6.0 / 40.0
    egp = 0.0161 * 75.0 * (1.0 - 0.03)
    renal = 0.0
    f01_corrected = 0.0097 * 75.0
    expected_dq1 = (
        u_id + egp - renal - f01_corrected - 0.01 * 96.0 + 0.066 * 20.0
    )
    expected_dq2 = 0.01 * 96.0 - (0.066 + 0.02) * 20.0

    assert derivative[6] == pytest.approx(expected_dq1)
    assert derivative[7] == pytest.approx(expected_dq2)


def test_f01_scales_with_glucose_below_4_5_mmol_l() -> None:
    state = np.zeros(10, dtype=np.float64)
    state[6] = 4.0 * V_G_L_PER_KG * BODY_WEIGHT_KG
    parameters = _base_parameters(EGP_0=0.0, AG=0.0, k_12=0.0)

    derivative = hovorka_ode(0.0, state, 0.0, 0.0, 1.0, parameters)

    expected_f01 = 0.0097 * BODY_WEIGHT_KG * 4.0 / 4.5
    assert derivative[6] == pytest.approx(-expected_f01)


def test_f01_is_unscaled_at_5_mmol_l() -> None:
    state = np.zeros(10, dtype=np.float64)
    state[6] = 5.0 * V_G_L_PER_KG * BODY_WEIGHT_KG
    parameters = _base_parameters(EGP_0=0.0, AG=0.0, k_12=0.0)

    derivative = hovorka_ode(0.0, state, 0.0, 0.0, 1.0, parameters)

    expected_f01 = 0.0097 * BODY_WEIGHT_KG
    assert derivative[6] == pytest.approx(-expected_f01)


def test_external_glucose_input_is_grams_per_minute() -> None:
    state = np.zeros(10, dtype=np.float64)
    parameters = _base_parameters(EGP_0=0.0, F_01=0.0)

    derivative = hovorka_ode(
        t=0.0,
        y=state,
        u_i_min=0.0,
        u_g_g_min=1.0,
        f_sens=1.0,
        p=parameters,
    )

    assert derivative[8] == pytest.approx(1.0 / 0.180182)


def test_empty_config_uses_canonical_base_parameters() -> None:
    packed = pack_params({}, BODY_WEIGHT_KG)

    assert packed[2] == pytest.approx(0.03)
    assert packed[3:6] == pytest.approx([0.03072, 0.0492, 1.56])
    assert packed[13] == pytest.approx(0.8)
    assert packed[14] == pytest.approx(40.0)
    assert packed[15] == pytest.approx(0.0)


def test_initial_insulin_action_uses_the_packed_effective_parameters() -> None:
    patient = HovorkaPatientModel(
        PatientConfig(name="canonical-defaults", params={}, body_weight_kg=BODY_WEIGHT_KG)
    )
    packed = pack_params({}, BODY_WEIGHT_KG)
    compartments = patient.state.compartments

    insulin = compartments["I"]
    assert compartments["x1"] == pytest.approx(packed[3] / packed[0] * insulin)
    assert compartments["x2"] == pytest.approx(packed[4] / packed[1] * insulin)
    assert compartments["x3"] == pytest.approx(packed[5] / packed[2] * insulin)


def test_initial_q2_is_at_steady_state_for_the_base_transfer_equation() -> None:
    patient = HovorkaPatientModel(
        PatientConfig(name="canonical-defaults", params={}, body_weight_kg=BODY_WEIGHT_KG)
    )
    compartments = patient.state.compartments
    state = np.array(
        [
            compartments[name]
            for name in ("S1", "S2", "I", "x1", "x2", "x3", "Q1", "Q2", "D1", "D2")
        ]
    )

    derivative = hovorka_ode(
        0.0,
        state,
        1.0 / 60.0,
        0.0,
        1.0,
        pack_params({}, BODY_WEIGHT_KG),
    )

    assert derivative[7] == pytest.approx(0.0, abs=1e-12)
