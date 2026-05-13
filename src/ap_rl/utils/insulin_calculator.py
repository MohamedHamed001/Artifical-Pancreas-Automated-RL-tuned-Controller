"""Meal-bolus + correction-dose calculator.

Formulas follow the conventional clinical heuristics:

* ``TDI = body_weight_kg * 0.55``  (total daily insulin estimate)
* ``carb_ratio = 500 / TDI``        (grams of carbs covered per 1 U)
* ``ISF = 1500 / TDI``              (mg/dL drop per 1 U)

The original equations are preserved verbatim. The class adds
**optional overrides** so demos and synthetic profiles can tune
``carb_ratio`` and ``isf`` without changing the underlying math when
overrides are unset.
"""

from __future__ import annotations

from typing import Optional


class InsulinCalculator:
    """Calculate meal bolus + correction insulin doses."""

    def __init__(
        self,
        patient_weight_kg: float = 75.0,
        carb_ratio: Optional[float] = None,
        isf: Optional[float] = None,
        lockout_duration_min: float = 15.0,
    ) -> None:
        """Initialise the calculator.

        Args:
            patient_weight_kg: Body weight in kg; drives TDI baseline.
            carb_ratio: Optional override (g carbs per 1 U insulin). Set
                to ``None`` to compute from weight via ``500 / TDI``.
            isf: Optional override (mg/dL drop per 1 U). Set to ``None``
                to compute from weight via ``1500 / TDI``.
            lockout_duration_min: Minimum minutes between consecutive
                bolus deliveries.
        """
        self.patient_weight = patient_weight_kg
        self.tdi = self._calculate_tdi()
        self._carb_ratio_override = carb_ratio
        self._isf_override = isf
        self.carb_ratio = (
            carb_ratio if carb_ratio is not None else self._calculate_carb_ratio()
        )
        self.isf = isf if isf is not None else self._calculate_isf()

        self.last_insulin_time = 0
        self.current_time = 0
        self.insulin_lockout_duration = lockout_duration_min

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
        """Override the carb ratio (used by demo / synthetic profiles)."""
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
        """Correction dose (units). Negative when current below target."""
        glucose_difference = current_glucose_mgdl - target_glucose_mgdl
        return glucose_difference / self.isf

    def deliver_bolus(
        self,
        carbs_grams: float,
        current_glucose_mgdl: float,
        target_glucose_mgdl: float = 120.0,
    ) -> dict:
        """Compute and (logically) deliver a meal bolus + correction."""
        if self.is_insulin_locked_out():
            return {
                "bolus_dose": 0.0,
                "correction_dose": 0.0,
                "total_dose": 0.0,
                "delivered": False,
                "reason": "Insulin lockout active",
            }

        bolus_dose = self.calculate_meal_bolus(carbs_grams)
        correction_dose = self.calculate_correction_dose(
            current_glucose_mgdl, target_glucose_mgdl
        )
        total_dose = max(0.0, bolus_dose + correction_dose)

        self.last_insulin_time = self.current_time

        return {
            "bolus_dose": bolus_dose,
            "correction_dose": correction_dose,
            "total_dose": total_dose,
            "delivered": True,
            "reason": "Bolus delivered successfully",
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
        }

    def update_patient_weight(self, new_weight_kg: float) -> None:
        """Update weight, recomputing carb ratio + ISF unless overridden."""
        self.patient_weight = new_weight_kg
        self.tdi = self._calculate_tdi()
        if self._carb_ratio_override is None:
            self.carb_ratio = self._calculate_carb_ratio()
        if self._isf_override is None:
            self.isf = self._calculate_isf()
