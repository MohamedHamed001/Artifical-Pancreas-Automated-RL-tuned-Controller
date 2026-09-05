from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from ap_rl.core.types import PatientConfig
from ap_rl.simulation.patient import HovorkaPatientModel

class HovorkaPatient:
    """
    Compatibility wrapper for HovorkaPatientModel.
    Preserves the legacy API while using the stabilized simulation core.
    """

    def __init__(self, patient_params: dict, simulation_start_time: int = 0) -> None:
        self.params = patient_params
        self.BW = self.params.get("BW", 75.0)
        self.simulation_start_time = simulation_start_time

        config = PatientConfig(
            name=patient_params.get("name", "unknown"),
            params=patient_params,
            body_weight_kg=self.BW
        )
        self._model = HovorkaPatientModel(config)
        self._model.reset()

        self.meal_data: Optional[pd.DataFrame] = None
        self.exercise_data: Optional[pd.DataFrame] = None

        # Track time manually to support the legacy API
        self.time = simulation_start_time

    @property
    def G(self) -> float:
        """Current glucose in mmol/L (legacy expected unit)."""
        return self._model.state.glucose_mgdl / 18.0182

    @property
    def S1(self) -> float: return self._model.state.compartments["S1"]
    @property
    def S2(self) -> float: return self._model.state.compartments["S2"]
    @property
    def I(self) -> float: return self._model.state.compartments["I"]
    @property
    def x1(self) -> float: return self._model.state.compartments["x1"]
    @property
    def x2(self) -> float: return self._model.state.compartments["x2"]
    @property
    def x3(self) -> float: return self._model.state.compartments["x3"]
    @property
    def Q1(self) -> float: return self._model.state.compartments["Q1"]
    @property
    def Q2(self) -> float: return self._model.state.compartments["Q2"]
    @property
    def D1(self) -> float: return self._model.state.compartments["D1"]
    @property
    def D2(self) -> float: return self._model.state.compartments["D2"]

    def set_meal_data(self, data: list[dict] | pd.DataFrame) -> None:
        if isinstance(data, list):
            self.meal_data = pd.DataFrame(data)
        else:
            self.meal_data = data

    def set_exercise_data(self, data: list[dict] | pd.DataFrame) -> None:
        if isinstance(data, list):
            self.exercise_data = pd.DataFrame(data)
        else:
            self.exercise_data = data

    def _get_meal_intake(self, t: float) -> float:
        if self.meal_data is None:
            return 0.0
        # Legacy exact match
        meal_events = self.meal_data[self.meal_data["time"] == t]
        if not meal_events.empty:
            return float(meal_events["carbs"].iloc[0])
        return 0.0

    def _get_exercise_status(self, t: float) -> bool:
        if self.exercise_data is None:
            return False
        exercise_events = self.exercise_data[self.exercise_data["time"] == t]
        if not exercise_events.empty:
            return bool(exercise_events["active"].iloc[0])
        return False

    def step(self, insulin_rate: float) -> dict[str, float]:
        """Advance by 1 min. Legacy API."""
        meal = self._get_meal_intake(self.time)
        exercise = self._get_exercise_status(self.time)

        state = self._model.step(int(self.time), insulin_rate, meal, exercise)
        self.time += 1

        # Return dict matching legacy expectations
        res = state.compartments.copy()
        res["glucose"] = state.glucose_mgdl
        res["time"] = float(state.time_min)
        return res

    def reset(self) -> float:
        state = self._model.reset()
        self.time = self.simulation_start_time
        return state.glucose_mgdl
