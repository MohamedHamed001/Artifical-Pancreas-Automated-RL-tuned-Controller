"""Parse meal + exercise schedules from text files.

Two formats are supported:

* The verbose ``TestCases.txt`` layout (``Meal Time: 480 min, Carbs: 80 g``).
* The two-column ``MealData_caseN.data`` / ``ExerciseData_caseN.data``
  layout (whitespace-separated ``time<sp>value`` with optional ``#``
  comments). The two-column variant is handled directly by
  :class:`HovorkaPatient`; this parser focuses on the verbose layout
  used by ``TestCases.txt``.
"""

from __future__ import annotations

import os
import re

import pandas as pd


class MealParser:
    """Parse the verbose TestCases.txt scenario format."""

    def __init__(self) -> None:
        self.meal_data: pd.DataFrame | None = None
        self.exercise_data: pd.DataFrame | None = None

    def parse_test_case(self, test_case_file: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse a TestCases.txt file.

        Returns a ``(meal_df, exercise_df)`` tuple sorted by time.
        Each meal is a row of ``time`` (minutes, int) and ``carbs`` (g).
        Each exercise event is a row of ``time`` and ``active`` (0/1).
        """
        meals: list[dict] = []
        exercises: list[dict] = []

        try:
            with open(test_case_file, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Test case file {test_case_file} not found")
            return self._to_frames(meals, exercises)

        for line in lines:
            line = line.strip()
            if line.startswith("Meal Time:") or (
                "Time:" in line and "Carb" in line and "Meal" in line
            ):
                try:
                    parts = line.split(",")
                    time_section = parts[0].split(":", 1)[1].strip()
                    time_match = re.search(r"(\d+\.?\d*)", time_section)
                    carbs_section = parts[1].split(":", 1)[1].strip()
                    carbs_match = re.search(r"(\d+\.?\d*)", carbs_section)
                    if not (time_match and carbs_match):
                        continue
                    meals.append(
                        {
                            "time": int(float(time_match.group(1))),
                            "carbs": float(carbs_match.group(1)),
                        }
                    )
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse meal line: {line}")
                    continue
            elif line.startswith("Exercise") and "Start" in line and "Duration" in line:
                try:
                    parts = line.split(",")
                    start = int(parts[0].split("=")[1].strip().replace(" minutes", ""))
                    duration = int(parts[1].split("=")[1].strip().replace(" minutes", ""))
                    exercises.append({"time": start, "active": 1})
                    exercises.append({"time": start + duration, "active": 0})
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse exercise line: {line}")
                    continue

        return self._to_frames(meals, exercises)

    def _to_frames(
        self, meals: list[dict], exercises: list[dict]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        meal_df = (
            pd.DataFrame(meals).sort_values("time").reset_index(drop=True)
            if meals
            else pd.DataFrame(columns=["time", "carbs"])
        )
        exercise_df = (
            pd.DataFrame(exercises).sort_values("time").reset_index(drop=True)
            if exercises
            else pd.DataFrame(columns=["time", "active"])
        )
        self.meal_data = meal_df
        self.exercise_data = exercise_df
        return meal_df, exercise_df

    def save_to_data_files(
        self,
        meal_file_path: str | os.PathLike,
        exercise_file_path: str | os.PathLike,
    ) -> None:
        """Serialise parsed data into the two-column .data format."""
        if self.meal_data is not None:
            expanded: list[dict] = []
            for _, row in self.meal_data.iterrows():
                expanded.append({"time": row["time"], "carbs": row["carbs"]})
                expanded.append({"time": row["time"] + 1, "carbs": 0.0})
            with open(meal_file_path, "w") as f:
                f.write("# Time(min) Carbs(g)\n")
                for row in expanded:
                    f.write(f"{row['time']} {row['carbs']}\n")

        if self.exercise_data is not None:
            with open(exercise_file_path, "w") as f:
                f.write("# Time(min) Active(0/1)\n")
                for _, row in self.exercise_data.iterrows():
                    f.write(f"{row['time']} {row['active']}\n")

    def get_meal_summary(self) -> str:
        if self.meal_data is None or self.meal_data.empty:
            return "No meals found"
        summary = f"Found {len(self.meal_data)} meals:\n"
        for _, row in self.meal_data.iterrows():
            summary += f"  - Time: {row['time']} min, Carbs: {row['carbs']} g\n"
        return summary

    def get_exercise_summary(self) -> str:
        if self.exercise_data is None or self.exercise_data.empty:
            return "No exercise sessions found"
        sessions: list[dict] = []
        current: dict | None = None
        for _, row in self.exercise_data.iterrows():
            if row["active"] == 1:
                current = {"start": row["time"]}
            elif row["active"] == 0 and current is not None:
                current["end"] = row["time"]
                current["duration"] = current["end"] - current["start"]
                sessions.append(current)
                current = None
        summary = f"Found {len(sessions)} exercise sessions:\n"
        for i, session in enumerate(sessions, 1):
            summary += (
                f"  - Session {i}: Start: {session['start']} min, "
                f"Duration: {session['duration']} min\n"
            )
        return summary
