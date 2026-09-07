from pathlib import Path

import numpy as np
import pytest
import yaml

from ap_rl.simulation import hovorka


BODY_WEIGHT_KG = 75.0


@pytest.mark.parametrize(
    ("time_min", "amplitude", "phase_min", "expected"),
    [
        (0.0, 0.0, -60.0, 1.0),
        (300.0, 0.05, -60.0, 1.05),
        (1020.0, 0.05, -60.0, 0.95),
    ],
)
def test_circadian_multiplier_reaches_the_expected_daily_extrema(
    time_min: float,
    amplitude: float,
    phase_min: float,
    expected: float,
) -> None:
    assert hovorka.circadian_multiplier(time_min, amplitude, phase_min) == pytest.approx(
        expected
    )


def test_zero_amplitude_leaves_the_complete_rhs_unchanged() -> None:
    state = np.array(
        [0.8, 0.7, 0.02, 0.01, 0.02, 0.03, 96.0, 20.0, 4.0, 6.0],
        dtype=np.float64,
    )
    base_parameters = hovorka.pack_params({}, BODY_WEIGHT_KG)
    disabled_extension_parameters = hovorka.pack_params(
        {"A_EGP": 0.0, "phi_EGP": 360.0}, BODY_WEIGHT_KG
    )

    base_rhs = hovorka.hovorka_ode(300.0, state, 0.01, 0.0, 1.0, base_parameters)
    disabled_extension_rhs = hovorka.hovorka_ode(
        300.0, state, 0.01, 0.0, 1.0, disabled_extension_parameters
    )

    np.testing.assert_allclose(disabled_extension_rhs, base_rhs)


def test_circadian_profile_contains_only_the_egp_extension_parameters() -> None:
    profile_path = (
        Path(__file__).parents[2] / "configs" / "profiles" / "project_circadian.yaml"
    )

    with profile_path.open() as profile_file:
        profile = yaml.safe_load(profile_file)

    assert profile == {"patient_params": {"A_EGP": 0.05, "phi_EGP": -60}}
