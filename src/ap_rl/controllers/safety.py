from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from ap_rl.core.types import PatientState, InsulinCommand, SafetyDecision

class SafetySupervisor:
    """
    A transparent safety layer that sits between the controller and the environment.
    Enforces clinical constraints (e.g., maximum insulin stacking, low glucose suspension).

    This version is refactored to use standard core types.
    """

    def __init__(
        self,
        min_glucose_mgdl: float = 70.0,
        max_iob_factor: float = 3.0,
        enable_low_suspend: bool = True,
    ):
        self.min_glucose_mgdl = min_glucose_mgdl
        self.max_iob_factor = max_iob_factor  # Max IOB as multiple of hourly basal
        self.enable_low_suspend = enable_low_suspend

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
        requested_total = command.total_u_h
        delivered_total = requested_total
        events = []
        is_modified = False

        current_glucose = state.glucose_mgdl
        glucose_velocity = state.glucose_rate_mgdl_min
        iob = state.iob_u

        # 1. Low Glucose Suspend (LGS)
        if self.enable_low_suspend:
            # Simple 15-min projection
            predicted_15min = current_glucose + (glucose_velocity * 15)

            if current_glucose < self.min_glucose_mgdl:
                events.append("LGS_ACTIVE_HYPO")
                delivered_total = 0.0
                is_modified = True
            elif predicted_15min < self.min_glucose_mgdl:
                events.append("LGS_ACTIVE_PREDICTED_HYPO")
                delivered_total = 0.0
                is_modified = True

        # 2. IOB Stacking Protection (Clamp to basal if IOB is too high)
        if not is_modified:
            max_allowed_iob = basal_rate_uh * self.max_iob_factor
            if iob >= max_allowed_iob:
                if requested_total > basal_rate_uh:
                    events.append("IOB_CLAMP_ACTIVE")
                    delivered_total = min(requested_total, basal_rate_uh)
                    is_modified = True

        # 3. Dynamic Braking (Reduce basal if falling fast)
        if not is_modified:
            if glucose_velocity < -2.0:
                if delivered_total > basal_rate_uh * 0.5:
                    events.append("DYNAMIC_BRAKING_ACTIVE")
                    delivered_total = min(delivered_total, basal_rate_uh * 0.5)
                    is_modified = True

        delivered_command = InsulinCommand(
            time_min=command.time_min,
            basal_u_h=delivered_total,
            bolus_u=command.bolus_u,
            total_u_h=delivered_total,
            reason="; ".join(events) if events else "NORMAL"
        )

        return SafetyDecision(
            requested=command,
            delivered=delivered_command,
            is_modified=is_modified,
            active_constraints=events,
            explanation="; ".join(events) if events else "NORMAL",
            metadata={"events": events}
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return metadata about the current safety state."""
        return {
            "min_glucose_mgdl": self.min_glucose_mgdl,
            "max_iob_factor": self.max_iob_factor,
            "low_suspend_active": self.enable_low_suspend,
        }
