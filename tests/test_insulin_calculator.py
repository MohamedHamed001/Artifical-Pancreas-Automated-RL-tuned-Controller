"""Tests for the InsulinCalculator (carb-ratio + ISF math + overrides)."""

from __future__ import annotations

import math

import pytest

from ap_rl.utils.insulin_calculator import InsulinCalculator


@pytest.fixture
def calc() -> InsulinCalculator:
    return InsulinCalculator(patient_weight_kg=75.0)


def test_tdi_rule(calc: InsulinCalculator) -> None:
    assert calc.tdi == pytest.approx(75.0 * 0.55)


def test_carb_ratio_rule(calc: InsulinCalculator) -> None:
    assert calc.carb_ratio == pytest.approx(500.0 / calc.tdi)


def test_isf_rule(calc: InsulinCalculator) -> None:
    assert calc.isf == pytest.approx(1500.0 / calc.tdi)


def test_meal_bolus_zero_carbs(calc: InsulinCalculator) -> None:
    assert calc.calculate_meal_bolus(0) == 0.0


def test_meal_bolus_positive(calc: InsulinCalculator) -> None:
    assert calc.calculate_meal_bolus(60) == pytest.approx(60.0 / calc.carb_ratio)


def test_correction_dose_positive(calc: InsulinCalculator) -> None:
    dose = calc.calculate_correction_dose(180.0, target_glucose_mgdl=120.0)
    assert dose == pytest.approx((180.0 - 120.0) / calc.isf)


def test_correction_dose_negative(calc: InsulinCalculator) -> None:
    dose = calc.calculate_correction_dose(90.0, target_glucose_mgdl=120.0)
    assert dose < 0


def test_deliver_bolus_locks_out_subsequent_call(calc: InsulinCalculator) -> None:
    calc.set_current_time(100)
    first = calc.deliver_bolus(50, 180.0)
    assert first["delivered"] is True
    calc.set_current_time(105)
    second = calc.deliver_bolus(50, 180.0)
    assert second["delivered"] is False
    assert "lockout" in second["reason"].lower()


def test_deliver_bolus_after_lockout_succeeds(calc: InsulinCalculator) -> None:
    calc.set_current_time(0)
    calc.deliver_bolus(50, 180.0)
    calc.set_current_time(60)
    second = calc.deliver_bolus(50, 180.0)
    assert second["delivered"] is True


def test_carb_ratio_override_preserved_on_weight_update() -> None:
    calc = InsulinCalculator(patient_weight_kg=75.0, carb_ratio=10.0)
    assert calc.carb_ratio == 10.0
    calc.update_patient_weight(90.0)
    assert calc.carb_ratio == 10.0, "override must survive weight changes"


def test_isf_override_preserved_on_weight_update() -> None:
    calc = InsulinCalculator(patient_weight_kg=75.0, isf=50.0)
    assert calc.isf == 50.0
    calc.update_patient_weight(60.0)
    assert calc.isf == 50.0


def test_setters_apply_overrides(calc: InsulinCalculator) -> None:
    calc.set_carb_ratio(8.0)
    calc.set_isf(42.0)
    assert calc.carb_ratio == 8.0
    assert calc.isf == 42.0
    # Subsequent weight updates should not clobber explicit overrides.
    calc.update_patient_weight(100.0)
    assert calc.carb_ratio == 8.0
    assert calc.isf == 42.0


def test_total_dose_is_non_negative_when_negative_correction(calc: InsulinCalculator) -> None:
    calc.set_current_time(0)
    result = calc.deliver_bolus(carbs_grams=5, current_glucose_mgdl=60.0)
    # Bolus dose alone is positive, but correction is strongly negative;
    # the implementation clamps total to non-negative.
    assert result["total_dose"] >= 0.0
    assert math.isfinite(result["total_dose"])
