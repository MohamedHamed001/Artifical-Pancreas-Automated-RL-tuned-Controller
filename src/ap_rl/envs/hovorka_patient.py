"""Hovorka virtual-patient ODE simulator.

This is a clean repackaging of the original ``working_virtual_patient.py``
module. Scientific behaviour (parameter dictionary, equations, numerical
integrator) is preserved verbatim; only imports, paths, and type
annotations were touched.
"""

from __future__ import annotations

import os
import random
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

try:
    from numba import njit  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    NUMBA_AVAILABLE = False


class HovorkaPatient:
    """Type-1 Diabetes Hovorka-model simulator.

    Simulates glucose-insulin dynamics in response to meals, exercise and
    insulin infusion. One ``step`` advances the simulation by 1 minute.

    Args:
        patient_params: dict of Hovorka parameters (see
            ``configs/patient_default.yaml`` for keys and defaults).
        simulation_start_time: simulation clock at t=0 (minutes).
    """

    def __init__(self, patient_params: dict, simulation_start_time: int = 0) -> None:
        self.params = patient_params
        self.BW = self.params.get("BW", 75)

        self.S1 = 0.0
        self.S2 = 0.0
        self.I = 0.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0
        self.Q1 = 0.0
        self.Q2 = 0.0
        self.D1 = 0.0
        self.D2 = 0.0

        self.simulation_start_time = simulation_start_time
        self.time = self.simulation_start_time

        self.G = self.params.get("G_init", 10.0)
        V_G = self.params.get("V_G", 0.16) * self.BW
        self.Q1 = self.G * V_G

        self.initial_state = self._get_full_state()

        self.meal_data: Optional[pd.DataFrame] = None
        self.exercise_data: Optional[pd.DataFrame] = None

        self.last_meal_input = 0.0

        self.exercise_start_time: Optional[float] = None
        self.exercise_end_time: Optional[float] = None
        self.current_exercise_status = 0
        self.F_sensitivity = 1.0

        self._numba_params = np.array(
            [
                self.params.get("k_a1", 0.006),
                self.params.get("k_a2", 0.06),
                self.params.get("k_a3", 0.05),
                self.params.get("k_b1", 0.003),
                self.params.get("k_b2", 0.06),
                self.params.get("k_b3", 0.04),
                self.params.get("V_I", 0.12) * self.BW,
                self.params.get("t_max_I", 55),
                self.params.get("k_e", 0.138),
                self.params.get("F_01", 0.0097) * self.BW,
                self.params.get("V_G", 0.16) * self.BW,
                self.params.get("k_12", 0.066),
                self.params.get("EGP_0", 0.0161) * self.BW,
                self.params.get("AG", 1.0),
                self.params.get("t_max_G", 30),
                self.params.get("A_EGP", 0.05),
                self.params.get("phi_EGP", -60),
                self.params.get("G_thresh", 9.0),
                self.params.get("k_R", 0.0031),
            ],
            dtype=np.float64,
        )

        if NUMBA_AVAILABLE:
            _state_dummy = np.zeros(10, dtype=np.float64)
            _ = hovorka_one_min_step(_state_dummy, 0.0, 1.0, self._numba_params)

    def _get_full_state(self) -> np.ndarray:
        return np.array(
            [
                self.S1,
                self.S2,
                self.I,
                self.x1,
                self.x2,
                self.x3,
                self.Q1,
                self.Q2,
                self.D1,
                self.D2,
            ]
        )

    def _set_full_state(self, state: np.ndarray) -> None:
        (
            self.S1,
            self.S2,
            self.I,
            self.x1,
            self.x2,
            self.x3,
            self.Q1,
            self.Q2,
            self.D1,
            self.D2,
        ) = state

    def load_meal_data(self, filepath: str) -> None:
        """Parse a two-column meal data file into ``self.meal_data``.

        Treats every monotonic carb increase as a new meal start so the
        accumulating ``TestData`` format is correctly disambiguated.
        """
        try:
            raw_df = pd.read_csv(
                filepath, sep=r"\s+", header=None, names=["time", "carbs"], comment="#"
            )
            if raw_df["time"].max() <= 24:
                raw_df["time"] *= 60

            processed_meals: list[dict] = []
            last_carbs = 0.0
            for _, row in raw_df.iterrows():
                current_carbs = row["carbs"]
                if current_carbs > 0 and (last_carbs == 0 or current_carbs > last_carbs):
                    processed_meals.append({"time": row["time"], "carbs": current_carbs})
                last_carbs = current_carbs

            if not processed_meals:
                self.meal_data = pd.DataFrame(columns=["time", "carbs"])
            else:
                self.meal_data = pd.DataFrame(processed_meals)
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            self.meal_data = pd.DataFrame(columns=["time", "carbs"])

    def load_exercise_data(self, filepath: str) -> None:
        """Parse a two-column exercise data file into ``self.exercise_data``."""
        try:
            df = pd.read_csv(
                filepath, sep=r"\s+", header=None, names=["time", "active"], comment="#"
            )
            if df["time"].max() <= 24:
                df["time"] *= 60
            self.exercise_data = df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            self.exercise_data = None

    def _get_meal_intake(self, t: float) -> float:
        if self.meal_data is None:
            return 0.0
        meal_events = self.meal_data[self.meal_data["time"] == t]
        if not meal_events.empty:
            return float(meal_events["carbs"].iloc[0])
        return 0.0

    def _get_exercise_status(self, t: float) -> float:
        if self.exercise_data is None:
            return 0.0
        exercise_events = self.exercise_data[self.exercise_data["time"] == t]
        if not exercise_events.empty:
            return float(exercise_events["active"].iloc[0])
        return 0.0

    def _update_exercise_sensitivity(self, t: float) -> None:
        """Update the F_sensitivity multiplier with rise/decay around exercise."""
        exercise_status = self._get_exercise_status(t)

        if exercise_status == 1 and self.current_exercise_status == 0:
            self.exercise_start_time = t
            self.exercise_end_time = None
        elif exercise_status == 0 and self.current_exercise_status == 1:
            self.exercise_end_time = t

        self.current_exercise_status = exercise_status

        F_peak = self.params.get("F_peak", 1.35)
        K_rise = self.params.get("K_rise", 5.0)
        K_decay = self.params.get("K_decay", 0.01)

        if exercise_status == 1 and self.exercise_start_time is not None:
            t_rise = t - self.exercise_start_time
            self.F_sensitivity = 1 + (F_peak - 1) * (1 - np.exp(-K_rise * t_rise))
        elif exercise_status == 0 and self.exercise_end_time is not None:
            t_decay = t - self.exercise_end_time
            self.F_sensitivity = 1 + (F_peak - 1) * np.exp(-K_decay * t_decay)
        else:
            self.F_sensitivity = 1.0

    def _hovorka_model_equations(self, t: float, y: np.ndarray, u_I: float) -> list[float]:
        """ODE right-hand side for the 10-state Hovorka model."""
        S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2 = y

        k_a1 = self.params.get("k_a1", 0.006)
        k_a2 = self.params.get("k_a2", 0.06)
        k_a3 = self.params.get("k_a3", 0.05)
        k_b1 = self.params.get("k_b1", 0.003)
        k_b2 = self.params.get("k_b2", 0.06)
        k_b3 = self.params.get("k_b3", 0.04)
        V_I = self.params.get("V_I", 0.12) * self.BW
        t_max_I = self.params.get("t_max_I", 55)
        k_e = self.params.get("k_e", 0.138)

        F_01 = self.params.get("F_01", 0.0097) * self.BW
        V_G = self.params.get("V_G", 0.16) * self.BW
        k_12 = self.params.get("k_12", 0.066)

        EGP_0 = self.params.get("EGP_0", 0.0161) * self.BW

        AG = self.params.get("AG", 1.0)
        t_max_G = self.params.get("t_max_G", 40)

        dS1 = u_I - (S1 / t_max_I)
        dS2 = (S1 - S2) / t_max_I
        dI = (S2 / (t_max_I * V_I)) - k_e * I

        dx1 = k_b1 * I - k_a1 * x1
        dx2 = k_b2 * I - k_a2 * x2
        dx3 = k_b3 * I - k_a3 * x3

        A_EGP = self.params.get("A_EGP", 0.05)
        phi_EGP = self.params.get("phi_EGP", -60)
        EGP0_baseline = EGP_0
        EGP0_circadian = EGP0_baseline * (
            1 + A_EGP * np.sin(2 * np.pi * (t - phi_EGP) / 1440)
        )
        EGP = EGP0_circadian - x3 * EGP_0

        G = Q1 / V_G if V_G > 0 else 0.0
        G_thresh = self.params.get("G_thresh", 9.0)
        k_R = self.params.get("k_R", 0.0031)
        if G > G_thresh:
            F_R = k_R * (G - G_thresh) * V_G
        else:
            F_R = 0.0

        dD1 = -D1 / t_max_G
        dD2 = D1 / t_max_G - D2 / t_max_G
        U_id = (AG * D2) / t_max_G

        U_g = x1 * Q1 * self.F_sensitivity

        dQ1 = U_id + EGP - F_R - F_01 - U_g - k_12 * Q1 + k_12 * Q2
        dQ2 = k_12 * Q1 - k_12 * Q2 - x2 * Q2

        return [dS1, dS2, dI, dx1, dx2, dx3, dQ1, dQ2, dD1, dD2]

    def step(self, insulin_rate: float) -> float:
        """Advance the simulation by one minute and return BGL (mg/dL)."""
        insulin_infusion_rate_umin = insulin_rate / 60.0

        self._update_exercise_sensitivity(self.time)

        meal_carbs = self._get_meal_intake(self.time)
        if meal_carbs > 0 and self.last_meal_input == 0:
            carbs_mmol = meal_carbs * (1000.0 / 180.0)
            self.D1 += carbs_mmol

        self.last_meal_input = meal_carbs

        if NUMBA_AVAILABLE:
            state = self._get_full_state().astype(np.float64)
            new_state = hovorka_one_min_step(
                state, insulin_infusion_rate_umin, self.F_sensitivity, self._numba_params
            )
            self._set_full_state(new_state)
        else:
            solution = solve_ivp(
                fun=lambda t, y: self._hovorka_model_equations(
                    t, y, insulin_infusion_rate_umin
                ),
                t_span=[self.time, self.time + 1],
                y0=self._get_full_state(),
                method="RK45",
            )
            self._set_full_state(solution.y[:, -1])

        V_G = self.params.get("V_G", 0.16) * self.BW
        self.G = self.Q1 / V_G

        self.time += 1

        return self.G * 18.0182

    def reset(self) -> float:
        """Reset the patient to its initial state and return initial BGL."""
        self._set_full_state(self.initial_state)
        self.time = self.simulation_start_time
        self.last_meal_input = 0.0
        self.exercise_start_time = None
        self.exercise_end_time = None
        self.current_exercise_status = 0
        self.F_sensitivity = 1.0
        return self.G * 18.0182


if NUMBA_AVAILABLE:

    @njit(fastmath=True, cache=True)  # type: ignore[misc]
    def hovorka_one_min_step(state, u_I, F_sens, p):
        """One-minute Euler integration of the Hovorka model (numba)."""
        S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2 = state

        (
            k_a1,
            k_a2,
            k_a3,
            k_b1,
            k_b2,
            k_b3,
            V_I,
            t_max_I,
            k_e,
            F_01,
            V_G,
            k_12,
            EGP_0,
            AG,
            t_max_G,
            A_EGP,
            phi_EGP,
            G_thresh,
            k_R,
        ) = p

        dS1 = u_I - (S1 / t_max_I)
        dS2 = (S1 - S2) / t_max_I
        dI = (S2 / (t_max_I * V_I)) - k_e * I

        dx1 = k_b1 * I - k_a1 * x1
        dx2 = k_b2 * I - k_a2 * x2
        dx3 = k_b3 * I - k_a3 * x3

        EGP = EGP_0 - x3 * EGP_0

        G = Q1 / V_G if V_G > 0.0 else 0.0
        if G > G_thresh:
            F_R = k_R * (G - G_thresh) * V_G
        else:
            F_R = 0.0

        dD1 = -D1 / t_max_G
        dD2 = D1 / t_max_G - D2 / t_max_G
        U_id = (AG * D2) / t_max_G

        U_g = x1 * Q1 * F_sens
        dQ1 = U_id + EGP - F_R - F_01 - U_g - k_12 * Q1 + k_12 * Q2
        dQ2 = k_12 * Q1 - k_12 * Q2 - x2 * Q2

        return np.array(
            [
                S1 + dS1,
                S2 + dS2,
                I + dI,
                x1 + dx1,
                x2 + dx2,
                x3 + dx3,
                Q1 + dQ1,
                Q2 + dQ2,
                D1 + dD1,
                D2 + dD2,
            ]
        )
