"""Meal-bolus + correction-dose calculator.

Formulas follow conventional clinical heuristics:

* ``TDI = body_weight_kg * 0.55``  (total daily insulin estimate)
* ``carb_ratio = 500 / TDI``        (grams of carbs covered per 1 U)
* ``ISF = 1500 / TDI``              (mg/dL drop per 1 U)

**ISF-adaptive split ratio**:
Insulin-sensitive patients (high ISF) are at greater risk of post-bolus
hypoglycemia if the full bolus is delivered immediately.  The split ratio
(fraction delivered upfront) is now automatically computed as::

    split_ratio = clip(1 - ISF / 120, 0.25, 0.70)

This yields:
  ISF=36 (standard)  → split=0.70  (70% immediate, 30% tail)
  ISF=60 (sensitive) → split=0.50  (50% immediate, 50% tail)
  ISF=80 (very sens) → split=0.33  (33% immediate, 67% tail)

The tail duration also scales: ``tail_duration = 30 + ISF * 0.5`` minutes.

**Extended / split bolus** (dual-wave):
Call :meth:`drain_tail_dose` once per minute to receive the per-minute
tail delivery.  This eliminates the large post-meal spike caused by
delayed insulin absorption.

**Correction dead-band**:
Correction doses are only issued when BGL exceeds ``target + dead_band``
(default 20 mg/dL) to prevent micro-corrections that stack with IOB.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class InsulinCalculator:
    """Calculate meal bolus + correction insulin doses.

    Args:
        patient_weight_kg: Body weight in kg; drives TDI baseline.
        carb_ratio: Optional override (g carbs per 1 U insulin).
        isf: Optional override (mg/dL drop per 1 U).
        lockout_duration_min: Minimum minutes between consecutive boluses.
        split_ratio: Fraction of total bolus delivered immediately (0–1).
            The remainder is spread over ``tail_duration_min`` minutes.
        tail_duration_min: Minutes over which the tail portion is dripped.
    """

    def __init__(
        self,
        patient_weight_kg: float = 75.0,
        carb_ratio: Optional[float] = None,
        isf: Optional[float] = None,
        lockout_duration_min: float = 15.0,
        split_ratio: Optional[float] = None,
        tail_duration_min: Optional[float] = None,
        correction_dead_band: float = 20.0,
    ) -> None:
        self.patient_weight = patient_weight_kg
        self.tdi = self._calculate_tdi()
        self._carb_ratio_override = carb_ratio
        self._isf_override = isf
        self.carb_ratio = (
            carb_ratio if carb_ratio is not None else self._calculate_carb_ratio()
        )
        self.isf = isf if isf is not None else self._calculate_isf()

        self.last_insulin_time = float("-inf")
        self.current_time = 0
        self.insulin_lockout_duration = lockout_duration_min

        # ISF-adaptive split ratio: sensitive patients get a smaller immediate
        # fraction to reduce crash risk.  Can be overridden explicitly.
        if split_ratio is not None:
            self.split_ratio = float(np.clip(split_ratio, 0.0, 1.0))
        else:
            # ISF=36 → 0.70,  ISF=60 → 0.50,  ISF=80 → 0.33
            self.split_ratio = float(np.clip(1.0 - self.isf / 120.0, 0.25, 0.70))

        # ISF-adaptive tail duration: longer tail for more sensitive patients.
        if tail_duration_min is not None:
            self.tail_duration_min = max(1.0, float(tail_duration_min))
        else:
            self.tail_duration_min = max(20.0, min(60.0, 30.0 + self.isf * 0.5))

        # Correction dead-band: don't correct unless BGL is this many mg/dL
        # above target (prevents stacking micro-corrections on top of IOB).
        self.correction_dead_band = float(correction_dead_band)

        self.pending_tail_dose: float = 0.0   # total tail units remaining
        self.tail_rate_per_min: float = 0.0   # U / min during tail phase

    def _calculate_tdi(self) -> float:
        """Total Daily Insulin estimate (units)."""
        return self.patient_weight * 0.55

    def _calculate_carb_ratio(self) -> float:
        """Grams of carbs covered per 1 U (500 rule)."""
        return 500.0 / self.tdi

    def _calculate_isf(self) -> float:
        """Insulin sensitivity factor in mg/dL per 1 U (1500 rule)."""
        return 1500.0 / self.tdi

    def set_carb_ratio(self, carb_ratio: float) -> None:
        """Override the carb ratio."""
        self._carb_ratio_override = carb_ratio
        self.carb_ratio = carb_ratio

    def set_isf(self, isf: float) -> None:
        """Override the insulin sensitivity factor."""
        self._isf_override = isf
        self.isf = isf

    def set_current_time(self, time_minutes: float) -> None:
        self.current_time = time_minutes

    def is_insulin_locked_out(self) -> bool:
        return (self.current_time - self.last_insulin_time) < self.insulin_lockout_duration

    def calculate_meal_bolus(self, carbs_grams: float) -> float:
        """Bolus dose in units for ``carbs_grams`` grams of carbs."""
        if carbs_grams <= 0:
            return 0.0
        return carbs_grams / self.carb_ratio

    def calculate_correction_dose(
        self, current_glucose_mgdl: float, target_glucose_mgdl: float = 120.0
    ) -> float:
        """Correction dose (units).

        Returns 0 when positive BGL is within the dead-band (``target + dead_band``).
        If BGL is below target, returns a negative dose (debt) as expected by
        numerical validation tests. High-level delivery methods clip this to 0.
        """
        glucose_difference = current_glucose_mgdl - target_glucose_mgdl

        # Gating: only issue positive correction if BGL is meaningfully above target
        if 0.0 < glucose_difference < self.correction_dead_band:
            return 0.0

        return glucose_difference / self.isf

    def drain_tail_dose(self) -> float:
        """Return and consume the per-minute tail delivery (as U/h rate).

        Call once per simulator minute.  Returns 0 when the tail is exhausted.
        The return value is already scaled to U/h to be compatible with
        the bolus_rate field in the environment step loop.
        """
        if self.pending_tail_dose <= 0.0:
            return 0.0
        delivered_per_min = min(self.pending_tail_dose, self.tail_rate_per_min)
        self.pending_tail_dose = max(0.0, self.pending_tail_dose - delivered_per_min)
        return delivered_per_min * 60.0  # U/min → U/h

    def deliver_bolus(
        self,
        carbs_grams: float,
        current_glucose_mgdl: float,
        target_glucose_mgdl: float = 120.0,
    ) -> dict:
        """Compute and deliver a split meal bolus + correction.

        The immediate portion (``split_ratio``) is returned in
        ``immediate_dose`` as units to deliver during the current simulator
        minute.  The tail portion is stored internally and dispensed each
        minute via :meth:`drain_tail_dose`.
        """
        if self.is_insulin_locked_out():
            return {
                "bolus_dose": 0.0,
                "correction_dose": 0.0,
                "total_dose": 0.0,
                "immediate_dose": 0.0,
                "tail_dose": 0.0,
                "delivered": False,
                "reason": "Insulin lockout active",
            }

        bolus_dose = self.calculate_meal_bolus(carbs_grams)
        correction_dose = self.calculate_correction_dose(
            current_glucose_mgdl, target_glucose_mgdl
        )
        total_dose = max(0.0, bolus_dose + correction_dose)

        # Split into immediate + tail
        immediate_dose = total_dose * self.split_ratio
        tail_dose = total_dose * (1.0 - self.split_ratio)

        # Queue the tail (replaces any previous unfinished tail)
        self.pending_tail_dose = tail_dose
        self.tail_rate_per_min = tail_dose / self.tail_duration_min

        self.last_insulin_time = self.current_time

        return {
            "bolus_dose": bolus_dose,
            "correction_dose": correction_dose,
            "total_dose": total_dose,
            "immediate_dose": immediate_dose,
            "tail_dose": tail_dose,
            "delivered": True,
            "reason": "Split bolus delivered successfully",
            "carb_ratio": self.carb_ratio,
            "isf": self.isf,
        }

    def deliver_correction(
        self, current_glucose_mgdl: float, target_glucose_mgdl: float = 120.0
    ) -> dict:
        """Compute and (logically) deliver a correction-only dose."""
        if self.is_insulin_locked_out():
            return {
                "correction_dose": 0.0,
                "delivered": False,
                "reason": "Insulin lockout active",
            }

        correction_dose = self.calculate_correction_dose(
            current_glucose_mgdl, target_glucose_mgdl
        )
        if correction_dose <= 0:
            return {
                "correction_dose": 0.0,
                "delivered": False,
                "reason": "No correction needed",
            }

        self.last_insulin_time = self.current_time
        return {
            "correction_dose": correction_dose,
            "delivered": True,
            "reason": "Correction delivered successfully",
            "isf": self.isf,
        }

    def get_insulin_parameters(self) -> dict:
        return {
            "patient_weight": self.patient_weight,
            "tdi": self.tdi,
            "carb_ratio": self.carb_ratio,
            "isf": self.isf,
            "lockout_duration": self.insulin_lockout_duration,
            "split_ratio": self.split_ratio,
            "tail_duration_min": self.tail_duration_min,
            "pending_tail_dose": self.pending_tail_dose,
        }

    def update_patient_weight(self, new_weight_kg: float) -> None:
        """Update weight, recomputing carb ratio + ISF unless overridden."""
        self.patient_weight = new_weight_kg
        self.tdi = self._calculate_tdi()
        if self._carb_ratio_override is None:
            self.carb_ratio = self._calculate_carb_ratio()
        if self._isf_override is None:
            self.isf = self._calculate_isf()

    def reset(self) -> None:
        """Reset tail-dose state (call at episode start)."""
        self.pending_tail_dose = 0.0
        self.tail_rate_per_min = 0.0
        self.last_insulin_time = float("-inf")
        self.current_time = 0
