from __future__ import annotations
import math
from typing import Dict, Any, Optional

from ap_rl.core.types import PatientState, InsulinCommand

class ClinicalZoneReward:
    """
    Standard clinical zone-based reward function.
    Asymmetric penalties for hypoglycemia and hyperglycemia.
    """

    def __init__(self, target_mgdl: float = 110.0, demo_mode: bool = False):
        self.target_mgdl = target_mgdl
        self.demo_mode = demo_mode

    def calculate(self, state: PatientState, prev_state: Optional[PatientState] = None) -> float:
        """
        Calculate reward based on current glucose and velocity.
        """
        glucose = state.glucose_mgdl
        velocity = state.glucose_rate_mgdl_min

        # 1. Catastrophic Penalties
        if glucose < 40:
            return -10000.0
        if glucose > 500:
            return -3000.0

        reward = 0.0

        # 2. Zone Model
        if 70.0 <= glucose <= 180.0:
            # TIR zone - Gaussian pull toward 100 mg/dL
            distance = abs(glucose - 100.0)
            reward += 50.0 * math.exp(-(distance ** 2) / (2 * 35.0 ** 2))
        elif 54.0 <= glucose < 70.0:
            reward -= (70.0 - glucose) * 4.0
        elif glucose < 54.0:
            reward -= ((70.0 - glucose) ** 2) * 1.5
        elif 180.0 < glucose <= 250.0:
            reward -= (glucose - 180.0) * 1.2
        elif glucose > 250.0:
            reward -= ((glucose - 180.0) ** 1.8) * 0.15

        # 3. Rate of Change Penalty
        abs_rate = abs(velocity)
        if abs_rate > 3.0:
            reward -= (abs_rate - 3.0) ** 2 * 2.0
        elif abs_rate > 2.0:
            reward -= (abs_rate - 2.0) * 1.5

        # 4. Stability Bonus
        if abs(velocity) <= 1.5:
            reward += 8.0
        elif abs(velocity) <= 3.0:
            reward += 3.0

        return float(reward)
