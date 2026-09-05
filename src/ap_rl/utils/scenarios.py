from __future__ import annotations
import re
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from ap_rl.core.types import Scenario

class ScenarioLoader:
    """
    Loads legacy scenario data files (.data).
    Format is typically space-separated columns with a header:
    # Table format: 1D
    time   value
    """

    @staticmethod
    def load_data_file(path: str | Path) -> np.ndarray:
        """Loads a .data file into a numpy array [time, value]."""
        data = []
        if not os.path.exists(path):
            return np.zeros((0, 2))

        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])

        return np.array(data)

    @staticmethod
    def get_case_files(data_dir: str | Path, case_id: int) -> Dict[str, Path]:
        """Returns paths for meal and exercise files for a given case."""
        d = Path(data_dir)
        return {
            "meal": d / f"MealData_case{case_id}.data",
            "exercise": d / f"ExerciseData_case{case_id}.data"
        }

    @staticmethod
    def _parse_test_cases(data_dir: str | Path) -> Dict[int, Dict[str, Any]]:
        """Parses TestCases.txt to get structured meals, exercise, and weight."""
        cases = {}
        txt_path = Path(data_dir) / "TestCases.txt"
        if not txt_path.exists():
            return cases

        current_case = None
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                m = re.match(r"Test Case \[(\d+)\]", line)
                if m:
                    current_case = int(m.group(1))
                    cases[current_case] = {"weight": 75.0, "meals": [], "exercise": []}
                    continue

                if current_case is None: continue

                if line.startswith("Body Weight:"):
                    w = re.search(r"([\d.]+)\s*kg", line)
                    if w: cases[current_case]["weight"] = float(w.group(1))

                elif line.startswith("Meal ") and "Time:" in line:
                    m = re.search(r"Time:\s*(\d+)\s*minutes,\s*Carb Amount:\s*([\d.]+)", line)
                    if m:
                        cases[current_case]["meals"].append({
                            "time": int(m.group(1)),
                            "carbs": float(m.group(2))
                        })

                elif line.startswith("Exercise ") and "Start =" in line:
                    m = re.search(r"Start =\s*(\d+)\s*minutes,\s*Duration =\s*(\d+)", line)
                    if m:
                        cases[current_case]["exercise"].append({
                            "start": int(m.group(1)),
                            "duration": int(m.group(2))
                        })
        return cases

    @staticmethod
    def load_case(case_id: int, data_dir: Optional[str | Path] = None) -> Scenario:
        """Loads a full scenario case with structured events and lookup tables."""
        if data_dir is None:
            from ap_rl.utils.paths import data_dir as get_data_dir
            data_dir = get_data_dir()

        d = Path(data_dir)
        files = ScenarioLoader.get_case_files(d, case_id)
        meal_data = ScenarioLoader.load_data_file(files["meal"])
        exercise_data = ScenarioLoader.load_data_file(files["exercise"])

        # Parse structured events
        all_cases = ScenarioLoader._parse_test_cases(d)
        case_info = all_cases.get(case_id, {"meals": [], "exercise": []})

        return Scenario(
            id=f"case{case_id}",
            meals=case_info["meals"],
            exercise=case_info["exercise"],
            meal_data=meal_data,
            exercise_data=exercise_data,
            duration_min=1440
        )
