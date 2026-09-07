import numpy as np
import pytest

from ap_rl.core.types import PatientConfig
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.simulation.iob import RapidActingIOB
from ap_rl.simulation.simulator import SimulationConfig, SimulationRunner


class FixedController:
    def __init__(self, rate_u_h: float) -> None:
        self.rate_u_h = rate_u_h

    def get_action(self, observation, info=None):
        return np.array([self.rate_u_h])

    def reset(self) -> None:
        pass

    def update(self, reward, done) -> None:
        pass


def _make_runner(rate_u_h: float, *, params: dict | None = None) -> SimulationRunner:
    return SimulationRunner(
        SimulationConfig(
            patient_config=PatientConfig(
                name="iob-test",
                params=params or DEFAULT_PATIENT_PARAMS,
                body_weight_kg=75.0,
            ),
            controller=FixedController(rate_u_h),
            duration_min=1,
            dt_min=1,
        )
    )


def test_delivered_dose_raises_iob() -> None:
    model = RapidActingIOB(duration_min=240)

    assert model.advance(delivered_u=1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("duration_min", [0, -1, 120.5, True, False])
def test_invalid_duration_is_rejected(duration_min: object) -> None:
    with pytest.raises(ValueError, match="positive whole number of minutes"):
        RapidActingIOB(duration_min=duration_min)  # type: ignore[arg-type]


def test_single_dose_follows_two_stage_decay_to_exact_cutoff() -> None:
    model = RapidActingIOB()
    model.advance(delivered_u=1.0)

    for _ in range(60):
        iob_at_60 = model.advance(delivered_u=0.0)
    for _ in range(60):
        iob_at_120 = model.advance(delivered_u=0.0)
    for _ in range(120):
        iob_at_240 = model.advance(delivered_u=0.0)

    assert iob_at_60 == pytest.approx(0.7091206793574372)
    assert iob_at_120 == pytest.approx(0.34612517372764706)
    assert iob_at_240 == 0.0


def test_configured_duration_sets_action_peak_and_cutoff() -> None:
    model = RapidActingIOB(duration_min=120)
    remaining = [model.advance(delivered_u=1.0)]
    remaining.extend(model.advance(delivered_u=0.0) for _ in range(120))
    action_by_minute = [
        remaining[minute] - remaining[minute + 1] for minute in range(120)
    ]

    assert max(range(120), key=action_by_minute.__getitem__) == 30
    assert all(iob >= 0.0 for iob in remaining)
    assert remaining[-1] == 0.0


def test_negative_delivery_is_rejected() -> None:
    model = RapidActingIOB()

    with pytest.raises(ValueError, match="nonnegative"):
        model.advance(delivered_u=-0.01)


def test_reset_clears_all_active_insulin() -> None:
    model = RapidActingIOB()
    model.advance(delivered_u=2.0)

    model.reset()

    assert model.advance(delivered_u=0.0) == 0.0


def test_runner_iob_excludes_insulin_blocked_by_safety() -> None:
    runner = _make_runner(rate_u_h=60.0)
    runner.safety.min_glucose_mgdl = 1000.0

    record = runner.step()

    assert record.requested_insulin == pytest.approx(60.0)
    assert record.delivered_insulin == 0.0
    assert record.iob == 0.0


def test_runner_converts_safety_delivered_rate_to_active_units() -> None:
    runner = _make_runner(rate_u_h=6.0)

    record = runner.step()

    assert record.delivered_insulin == pytest.approx(6.0)
    assert record.iob == pytest.approx(0.1)


def test_hovorka_depot_is_diagnostic_not_controller_iob() -> None:
    state = _make_runner(rate_u_h=0.0).patient.state

    assert state.iob_u == 0.0
    assert state.compartments["sc_depot_insulin_u"] == pytest.approx(
        state.compartments["S1"] + state.compartments["S2"]
    )
