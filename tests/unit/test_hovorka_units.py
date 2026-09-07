import pytest

from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.simulation.hovorka import pack_params


def test_runtime_kb_values_are_for_u_per_litre() -> None:
    params = DEFAULT_PATIENT_PARAMS
    packed = pack_params(params, body_weight=75.0)

    assert packed[3:6] == pytest.approx([0.03072, 0.0492, 1.56])
