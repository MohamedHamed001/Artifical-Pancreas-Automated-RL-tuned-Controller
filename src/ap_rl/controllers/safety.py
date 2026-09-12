from __future__ import annotations
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

from ap_rl.core.types import PatientState, InsulinCommand, SafetyDecision


_RATE_ROUNDOFF_TOLERANCE_U_H = 1e-12


@dataclass(frozen=True)
class SafetyPolicy:
    min_glucose_mgdl: float = 70.0
    prediction_horizon_min: float = 15.0
    max_iob_factor: float = 3.0
    falling_glucose_threshold_mgdl_min: float = -2.0
    braking_rate_factor: float = 0.5
    max_delivery_rate_u_h: float = 10.0
    enable_low_suspend: bool = True

    def __post_init__(self) -> None:
        numeric_fields = (
            "min_glucose_mgdl",
            "prediction_horizon_min",
            "max_iob_factor",
            "falling_glucose_threshold_mgdl_min",
            "braking_rate_factor",
            "max_delivery_rate_u_h",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            try:
                finite = not isinstance(value, bool) and math.isfinite(value)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{field_name} must be finite")

        for field_name in (
            "min_glucose_mgdl",
            "prediction_horizon_min",
            "max_iob_factor",
            "max_delivery_rate_u_h",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")

        if not 0.0 < self.braking_rate_factor <= 1.0:
            raise ValueError("braking_rate_factor must be in (0.0, 1.0]")


class SafetySupervisor:
    """
    A transparent safety layer that sits between the controller and the environment.
    Enforces clinical constraints (e.g., maximum insulin stacking, low glucose suspension).

    This version is refactored to use standard core types.
    """

    def __init__(
        self,
        policy: SafetyPolicy | None = None,
        *,
        min_glucose_mgdl: float | None = None,
        max_iob_factor: float | None = None,
        enable_low_suspend: bool | None = None,
    ):
        self.policy = policy or SafetyPolicy()
        overrides = {
            name: value
            for name, value in {
                "min_glucose_mgdl": min_glucose_mgdl,
                "max_iob_factor": max_iob_factor,
                "enable_low_suspend": enable_low_suspend,
            }.items()
            if value is not None
        }
        if overrides:
            self.policy = replace(self.policy, **overrides)

    @property
    def min_glucose_mgdl(self) -> float:
        return self.policy.min_glucose_mgdl

    @min_glucose_mgdl.setter
    def min_glucose_mgdl(self, value: float) -> None:
        self.policy = replace(self.policy, min_glucose_mgdl=value)

    @property
    def max_iob_factor(self) -> float:
        return self.policy.max_iob_factor

    @max_iob_factor.setter
    def max_iob_factor(self, value: float) -> None:
        self.policy = replace(self.policy, max_iob_factor=value)

    @property
    def enable_low_suspend(self) -> bool:
        return self.policy.enable_low_suspend

    @enable_low_suspend.setter
    def enable_low_suspend(self, value: bool) -> None:
        self.policy = replace(self.policy, enable_low_suspend=value)

    def evaluate(
        self,
        command: InsulinCommand,
        state: PatientState,
        basal_rate_uh: float,
    ) -> SafetyDecision:
        """
        Evaluate and potentially modify the insulin dose based on safety rules.

        Args:
            command: The requested insulin command from a controller.
            state: The current patient state.
            basal_rate_uh: The patient's reference hourly basal rate (U/h).

        Returns:
            A SafetyDecision containing the safe dose and metadata.
        """
        finite_inputs = {
            "command.basal_u_h": command.basal_u_h,
            "command.bolus_u": command.bolus_u,
            "command.total_u_h": command.total_u_h,
            "state.glucose_mgdl": state.glucose_mgdl,
            "state.glucose_rate_mgdl_min": state.glucose_rate_mgdl_min,
            "state.iob_u": state.iob_u,
            "basal_rate_uh": basal_rate_uh,
        }
        for name, value in finite_inputs.items():
            try:
                finite = not isinstance(value, bool) and math.isfinite(value)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")

        for name in (
            "command.basal_u_h",
            "command.total_u_h",
            "basal_rate_uh",
        ):
            if -_RATE_ROUNDOFF_TOLERANCE_U_H <= finite_inputs[name] < 0.0:
                finite_inputs[name] = 0.0

        for name in (
            "command.basal_u_h",
            "command.bolus_u",
            "command.total_u_h",
            "state.glucose_mgdl",
            "state.iob_u",
            "basal_rate_uh",
        ):
            if finite_inputs[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")

        requested_total = finite_inputs["command.total_u_h"]
        basal_rate_uh = finite_inputs["basal_rate_uh"]
        delivered_total = requested_total
        current_glucose = state.glucose_mgdl
        glucose_velocity = state.glucose_rate_mgdl_min
        iob = state.iob_u
        predicted = current_glucose + (
            glucose_velocity * self.policy.prediction_horizon_min
        )
        if not math.isfinite(predicted):
            raise ValueError("predicted_min_glucose_mgdl must be finite")

        events = []

        low_event = None
        if self.policy.enable_low_suspend:
            if current_glucose < self.policy.min_glucose_mgdl:
                low_event = "LGS_ACTIVE_HYPO"
            elif predicted < self.policy.min_glucose_mgdl:
                low_event = "LGS_ACTIVE_PREDICTED_HYPO"

        max_allowed_iob = None
        braking_limit = None
        if low_event is not None:
            if delivered_total > 0.0:
                delivered_total = 0.0
                events.append(low_event)
        else:
            max_allowed_iob = basal_rate_uh * self.policy.max_iob_factor
            braking_limit = basal_rate_uh * self.policy.braking_rate_factor
            if not math.isfinite(max_allowed_iob):
                raise ValueError("iob_limit_u must be finite")
            if not math.isfinite(braking_limit):
                raise ValueError("braking_limit_u_h must be finite")

            if iob >= max_allowed_iob and delivered_total > basal_rate_uh:
                delivered_total = basal_rate_uh
                events.append("IOB_CLAMP_ACTIVE")

            if (
                glucose_velocity
                < self.policy.falling_glucose_threshold_mgdl_min
                and delivered_total > braking_limit
            ):
                delivered_total = braking_limit
                events.append("DYNAMIC_BRAKING_ACTIVE")

            if delivered_total > self.policy.max_delivery_rate_u_h:
                delivered_total = self.policy.max_delivery_rate_u_h
                events.append("ABSOLUTE_RATE_CAP_ACTIVE")

        is_modified = delivered_total != requested_total

        delivered_command = InsulinCommand(
            time_min=command.time_min,
            basal_u_h=delivered_total,
            bolus_u=0.0 if is_modified else command.bolus_u,
            total_u_h=delivered_total,
            reason="; ".join(events) if events else "NORMAL"
        )

        return SafetyDecision(
            requested=command,
            delivered=delivered_command,
            is_modified=is_modified,
            active_constraints=events,
            predicted_min_glucose_mgdl=predicted,
            explanation="; ".join(events) if events else "NORMAL",
            metadata={
                "requested_rate_u_h": requested_total,
                "delivered_rate_u_h": delivered_total,
                "predicted_min_glucose_mgdl": predicted,
                "iob_limit_u": max_allowed_iob,
                "braking_limit_u_h": braking_limit,
                "absolute_rate_limit_u_h": self.policy.max_delivery_rate_u_h,
            },
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return metadata about the current safety state."""
        return asdict(self.policy)
