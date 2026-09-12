import pytest

from ap_rl.controllers.safety import SafetyPolicy, SafetySupervisor
from ap_rl.core.types import InsulinCommand, PatientState


def command(
    total_u_h: float = 5.0,
    *,
    basal_u_h: float | None = None,
    bolus_u: float = 0.0,
) -> InsulinCommand:
    return InsulinCommand(
        time_min=30,
        basal_u_h=total_u_h if basal_u_h is None else basal_u_h,
        bolus_u=bolus_u,
        total_u_h=total_u_h,
        reason="controller request",
    )


def state(
    *,
    glucose_mgdl: float = 140.0,
    glucose_rate_mgdl_min: float = 0.0,
    iob_u: float = 0.0,
) -> PatientState:
    return PatientState(
        time_min=30,
        glucose_mgdl=glucose_mgdl,
        glucose_rate_mgdl_min=glucose_rate_mgdl_min,
        iob_u=iob_u,
        cob_g=0.0,
        exercise_active=False,
        compartments={},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_glucose_mgdl": 0.0},
        {"prediction_horizon_min": float("inf")},
        {"max_iob_factor": -1.0},
        {"braking_rate_factor": 1.1},
        {"max_delivery_rate_u_h": 0.0},
    ],
)
def test_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SafetyPolicy(**kwargs)


def test_policy_is_frozen():
    policy = SafetyPolicy()

    with pytest.raises((AttributeError, TypeError)):
        policy.min_glucose_mgdl = 75.0


def test_legacy_supervisor_keywords_build_a_policy():
    supervisor = SafetySupervisor(min_glucose_mgdl=75.0, max_iob_factor=2.0)

    assert supervisor.policy.min_glucose_mgdl == 75.0
    assert supervisor.policy.max_iob_factor == 2.0


def test_legacy_supervisor_attributes_update_the_policy():
    supervisor = SafetySupervisor()

    supervisor.min_glucose_mgdl = 75.0

    assert supervisor.policy.min_glucose_mgdl == 75.0


def test_actual_low_suspend_is_terminal():
    decision = SafetySupervisor(SafetyPolicy(max_delivery_rate_u_h=1.0)).evaluate(
        command=command(total_u_h=5.0),
        state=state(glucose_mgdl=69.0, glucose_rate_mgdl_min=-3.0, iob_u=5.0),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == 0.0
    assert decision.active_constraints == ["LGS_ACTIVE_HYPO"]


def test_predicted_low_suspend_uses_policy_horizon_and_is_terminal():
    policy = SafetyPolicy(prediction_horizon_min=20.0, max_delivery_rate_u_h=1.0)

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=5.0),
        state=state(glucose_mgdl=80.0, glucose_rate_mgdl_min=-1.0, iob_u=5.0),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == 0.0
    assert decision.active_constraints == ["LGS_ACTIVE_PREDICTED_HYPO"]
    assert decision.predicted_min_glucose_mgdl == pytest.approx(60.0)


def test_high_iob_clamps_request_above_basal():
    decision = SafetySupervisor(SafetyPolicy(max_iob_factor=1.0)).evaluate(
        command=command(total_u_h=5.0),
        state=state(iob_u=1.0),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == pytest.approx(1.0)
    assert decision.active_constraints == ["IOB_CLAMP_ACTIVE"]


def test_fast_fall_brakes_to_policy_fraction_of_basal():
    policy = SafetyPolicy(
        falling_glucose_threshold_mgdl_min=-1.0,
        braking_rate_factor=0.25,
    )

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=5.0),
        state=state(glucose_rate_mgdl_min=-1.5),
        basal_rate_uh=2.0,
    )

    assert decision.delivered.total_u_h == pytest.approx(0.5)
    assert decision.active_constraints == ["DYNAMIC_BRAKING_ACTIVE"]


def test_high_iob_and_fast_fall_compose_limits():
    policy = SafetyPolicy(max_iob_factor=1.0, braking_rate_factor=0.5)

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=5.0),
        state=state(iob_u=1.0, glucose_rate_mgdl_min=-3.0),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == pytest.approx(0.5)
    assert decision.active_constraints == [
        "IOB_CLAMP_ACTIVE",
        "DYNAMIC_BRAKING_ACTIVE",
    ]


def test_absolute_rate_cap_applies_last():
    policy = SafetyPolicy(max_delivery_rate_u_h=2.0)

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=5.0),
        state=state(),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == pytest.approx(2.0)
    assert decision.active_constraints == ["ABSOLUTE_RATE_CAP_ACTIVE"]


def test_only_gates_that_change_the_rate_emit_events():
    policy = SafetyPolicy(max_iob_factor=1.0, max_delivery_rate_u_h=0.75)

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=0.5),
        state=state(iob_u=1.0, glucose_rate_mgdl_min=-3.0),
        basal_rate_uh=1.0,
    )

    assert decision.delivered.total_u_h == pytest.approx(0.5)
    assert decision.active_constraints == []
    assert decision.is_modified is False


def test_delivered_command_and_metadata_report_one_consistent_rate():
    policy = SafetyPolicy(
        prediction_horizon_min=10.0,
        max_iob_factor=2.0,
        braking_rate_factor=0.5,
        max_delivery_rate_u_h=4.0,
    )

    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=8.0, basal_u_h=3.0),
        state=state(glucose_mgdl=140.0, glucose_rate_mgdl_min=-1.0, iob_u=1.0),
        basal_rate_uh=2.0,
    )

    assert decision.delivered.basal_u_h == pytest.approx(4.0)
    assert decision.delivered.total_u_h == pytest.approx(4.0)
    assert decision.predicted_min_glucose_mgdl == pytest.approx(130.0)
    assert decision.metadata == {
        "requested_rate_u_h": 8.0,
        "delivered_rate_u_h": 4.0,
        "predicted_min_glucose_mgdl": 130.0,
        "iob_limit_u": 4.0,
        "braking_limit_u_h": 1.0,
        "absolute_rate_limit_u_h": 4.0,
    }


@pytest.mark.parametrize(
    ("policy", "patient_state", "expected_rate"),
    [
        pytest.param(SafetyPolicy(), state(glucose_mgdl=69.0), 0.0, id="low"),
        pytest.param(
            SafetyPolicy(max_delivery_rate_u_h=2.0),
            state(),
            2.0,
            id="capped",
        ),
    ],
)
def test_modified_delivery_zeroes_unmodeled_bolus_and_reports_one_rate(
    policy,
    patient_state,
    expected_rate,
):
    decision = SafetySupervisor(policy).evaluate(
        command=command(total_u_h=5.0, basal_u_h=3.0, bolus_u=1.5),
        state=patient_state,
        basal_rate_uh=1.0,
    )

    assert decision.requested.bolus_u == pytest.approx(1.5)
    assert decision.delivered.total_u_h == pytest.approx(expected_rate)
    assert decision.delivered.basal_u_h == pytest.approx(expected_rate)
    assert decision.delivered.bolus_u == 0.0


def test_unmodified_delivery_preserves_requested_bolus_for_audit():
    requested = command(total_u_h=1.0, basal_u_h=1.0, bolus_u=1.5)

    decision = SafetySupervisor().evaluate(
        command=requested,
        state=state(),
        basal_rate_uh=1.0,
    )

    assert decision.is_modified is False
    assert decision.delivered.total_u_h == pytest.approx(1.0)
    assert decision.delivered.basal_u_h == pytest.approx(1.0)
    assert decision.delivered.bolus_u == pytest.approx(1.5)


def test_terminal_low_suspend_skips_inapplicable_derived_limits():
    decision = SafetySupervisor(SafetyPolicy(max_iob_factor=1.0e308)).evaluate(
        command=command(total_u_h=5.0),
        state=state(glucose_mgdl=69.0),
        basal_rate_uh=2.0,
    )

    assert decision.delivered.total_u_h == 0.0
    assert decision.active_constraints == ["LGS_ACTIVE_HYPO"]
    assert decision.predicted_min_glucose_mgdl == pytest.approx(69.0)
    assert decision.metadata == {
        "requested_rate_u_h": 5.0,
        "delivered_rate_u_h": 0.0,
        "predicted_min_glucose_mgdl": 69.0,
        "iob_limit_u": None,
        "braking_limit_u_h": None,
        "absolute_rate_limit_u_h": 10.0,
    }


@pytest.mark.parametrize(
    ("unsafe_command", "unsafe_state", "basal_rate_uh"),
    [
        (command(total_u_h=float("nan")), state(), 1.0),
        (command(basal_u_h=float("inf")), state(), 1.0),
        (command(bolus_u=-0.1), state(), 1.0),
        (command(total_u_h=-0.1), state(), 1.0),
        (command(), state(glucose_mgdl=float("nan")), 1.0),
        (command(), state(glucose_mgdl=-1.0), 1.0),
        (command(), state(glucose_rate_mgdl_min=float("inf")), 1.0),
        (command(), state(iob_u=-0.1), 1.0),
        (command(), state(), float("nan")),
        (command(), state(), -0.1),
    ],
)
def test_evaluate_rejects_invalid_numeric_inputs(
    unsafe_command,
    unsafe_state,
    basal_rate_uh,
):
    with pytest.raises(ValueError):
        SafetySupervisor().evaluate(
            command=unsafe_command,
            state=unsafe_state,
            basal_rate_uh=basal_rate_uh,
        )


def test_rate_roundoff_is_normalized_but_material_negative_rate_is_rejected():
    decision = SafetySupervisor().evaluate(
        command=command(total_u_h=-8.0e-14, basal_u_h=-8.0e-14),
        state=state(),
        basal_rate_uh=-8.0e-14,
    )

    assert decision.delivered.total_u_h == 0.0
    assert decision.metadata["requested_rate_u_h"] == 0.0

    with pytest.raises(ValueError):
        SafetySupervisor().evaluate(
            command=command(total_u_h=-1.1e-12),
            state=state(),
            basal_rate_uh=1.0,
        )


def test_evaluate_rejects_overflowed_limit_from_finite_inputs():
    policy = SafetyPolicy(max_iob_factor=1.0e308)

    with pytest.raises(ValueError, match="iob_limit_u must be finite"):
        SafetySupervisor(policy).evaluate(
            command=command(),
            state=state(),
            basal_rate_uh=2.0,
        )


def test_diagnostics_expose_complete_policy_as_plain_values():
    policy = SafetyPolicy(
        min_glucose_mgdl=75.0,
        prediction_horizon_min=20.0,
        max_iob_factor=2.0,
        falling_glucose_threshold_mgdl_min=-1.5,
        braking_rate_factor=0.25,
        max_delivery_rate_u_h=8.0,
        enable_low_suspend=False,
    )

    assert SafetySupervisor(policy).get_diagnostics() == {
        "min_glucose_mgdl": 75.0,
        "prediction_horizon_min": 20.0,
        "max_iob_factor": 2.0,
        "falling_glucose_threshold_mgdl_min": -1.5,
        "braking_rate_factor": 0.25,
        "max_delivery_rate_u_h": 8.0,
        "enable_low_suspend": False,
    }
