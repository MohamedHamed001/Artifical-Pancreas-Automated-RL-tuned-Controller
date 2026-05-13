"""Tests for the MealParser TestCases.txt parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ap_rl.utils.meal_parser import MealParser


@pytest.fixture
def sample_case(tmp_path: Path) -> Path:
    content = """Test Case [1]
Body Weight: 80.00 kg
Simulation Time: 1440 minutes

Meal Count: 2
Meal 1 Time: 480 minutes, Carb Amount: 45.0 grams
Meal 2 Time: 780 minutes, Carb Amount: 60.0 grams

Exercise Sessions:
Exercise 1: Start = 300 minutes, Duration = 30 minutes
"""
    path = tmp_path / "TestCases.txt"
    path.write_text(content)
    return path


def test_parse_meals_and_exercise(sample_case: Path) -> None:
    parser = MealParser()
    meals, exercises = parser.parse_test_case(sample_case)

    assert isinstance(meals, pd.DataFrame)
    assert list(meals.columns) == ["time", "carbs"]
    assert len(meals) == 2
    assert meals.iloc[0].to_dict() == {"time": 480, "carbs": 45.0}
    assert meals.iloc[1].to_dict() == {"time": 780, "carbs": 60.0}

    assert isinstance(exercises, pd.DataFrame)
    assert list(exercises.columns) == ["time", "active"]
    # start + end events
    assert len(exercises) == 2
    assert exercises.iloc[0].to_dict() == {"time": 300, "active": 1}
    assert exercises.iloc[1].to_dict() == {"time": 330, "active": 0}


def test_parse_handles_missing_file(tmp_path: Path) -> None:
    parser = MealParser()
    missing = tmp_path / "does_not_exist.txt"
    meals, exercises = parser.parse_test_case(missing)
    assert meals.empty and exercises.empty


def test_summaries_handle_empty(sample_case: Path) -> None:
    parser = MealParser()
    assert parser.get_meal_summary() == "No meals found"
    assert parser.get_exercise_summary() == "No exercise sessions found"

    parser.parse_test_case(sample_case)
    assert "Found 2 meals" in parser.get_meal_summary()
    assert "Found 1 exercise sessions" in parser.get_exercise_summary()


def test_save_round_trip(sample_case: Path, tmp_path: Path) -> None:
    parser = MealParser()
    parser.parse_test_case(sample_case)
    meal_out = tmp_path / "meals.data"
    exercise_out = tmp_path / "exercise.data"
    parser.save_to_data_files(meal_out, exercise_out)

    meal_text = meal_out.read_text()
    assert "480.0 45.0" in meal_text or "480 45.0" in meal_text
    # synthesised end-of-meal sentinel one minute after the meal start
    assert "481.0 0.0" in meal_text or "481 0.0" in meal_text

    ex_text = exercise_out.read_text()
    assert "300 1" in ex_text
    assert "330 0" in ex_text
