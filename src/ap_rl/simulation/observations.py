from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from ap_rl.core.types import PatientState

class ObservationBuilder:
    """
    Constructs the 19-D observation vector required by the legacy A2C/PPO actors.
    """

    def __init__(
        self,
        target_mgdl: float = 120.0,
        patient_weight: float = 75.0,
        isf: float = 50.0,
        carb_ratio: float = 15.0,
        dt_min: int = 5
    ):
        self.target_mgdl = target_mgdl
        self.patient_weight = patient_weight
        self.isf = isf
        self.carb_ratio = carb_ratio
        self.dt_min = dt_min

    def build(
        self,
        state: PatientState,
        time_min: int,
        pid_gains: tuple[float, float, float],
        exercise_active: bool = False
    ) -> np.ndarray:
        current_glucose = state.glucose_mgdl

        hour_of_day = (time_min % 1440) / 60.0
        time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

        # Identity features (normalised)
        isf_norm = float(np.clip(self.isf / 100.0, 0.0, 1.5))
        cr_norm = float(np.clip(self.carb_ratio / 20.0, 0.0, 1.5))
        weight_norm = float(np.clip(self.patient_weight / 120.0, 0.0, 1.5))

        Kp, Ki, Kd = pid_gains

        # [glucose, rate, error, iterm, dterm, kp, ki, kd, time_since_meal, time_since_ins,
        #  ex, sin, cos, rate2, bolus_rem, iob, isf, cr, weight]
        obs = np.array([
            current_glucose / 400.0,
            state.glucose_rate_mgdl_min / 100.0,
            (self.target_mgdl - current_glucose) / 200.0,
            0.0, # ITerm
            0.0, # DTerm
            Kp / 10.0,
            Ki,
            Kd,
            1.0, # time_since_meal (placeholder)
            0.0, # time_since_ins (placeholder)
            1.0 if exercise_active else 0.0,
            time_sin,
            time_cos,
            state.glucose_rate_mgdl_min / 100.0,
            0.0, # bolus_remaining
            state.iob_u / 10.0,
            isf_norm,
            cr_norm,
            weight_norm
        ], dtype=np.float32)

        return obs
