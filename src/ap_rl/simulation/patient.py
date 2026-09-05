from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Protocol
from abc import ABC, abstractmethod

from ap_rl.core.types import PatientState, PatientConfig
from ap_rl.simulation.hovorka import hovorka_step, pack_params


class PatientModel(ABC):
    """Abstract base class for all virtual patient models."""

    @abstractmethod
    def reset(self, config: Optional[PatientConfig] = None) -> PatientState:
        """Reset patient to initial state."""
        pass

    @abstractmethod
    def step(self, t: int, insulin_rate_u_h: float, meal_carbs_g: float, exercise_active: bool) -> PatientState:
        """Advance simulation by 1 minute."""
        pass

    @property
    @abstractmethod
    def state(self) -> PatientState:
        """Get current patient state."""
        pass


class HovorkaPatientModel(PatientModel):
    """
    Implementation of the Hovorka patient model using the new simulation core.
    """

    def __init__(self, config: PatientConfig):
        self.config = config
        self._p_array = pack_params(config.params, config.body_weight_kg)
        self._state_vec = self._initialize_state()
        self._time_min = 0

        # Exercise sensitivity tracking
        self._f_sens = 1.0
        self._exercise_start_time: Optional[int] = None
        self._exercise_end_time: Optional[int] = None
        self._prev_exercise_active = False

    def _initialize_state(self) -> np.ndarray:
        """Set up initial 10-state vector with steady-state insulin assumptions."""
        # [S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2]
        state = np.zeros(10, dtype=np.float64)

        # Basal insulin assumption for initialization (U/h -> U/min)
        # We'll use 1.0 U/h as a safe default if not provided elsewhere
        basal_u_min = 1.0 / 60.0

        # Pull parameters
        params = self.config.params
        t_max_I = params.get("t_max_I", 55.0)
        V_I = params.get("V_I", 0.12) * self.config.body_weight_kg
        k_e = params.get("k_e", 0.138)

        # Insulin action rates
        k_a1 = params.get("k_a1", 0.006)
        k_a2 = params.get("k_a2", 0.06)
        k_a3 = params.get("k_a3", 0.05)
        k_b1 = params.get("k_b1", 0.003)
        k_b2 = params.get("k_b2", 0.06)
        k_b3 = params.get("k_b3", 0.04)

        # 1. Steady-state Insulin (Absorption sub-model)
        state[0] = basal_u_min * t_max_I  # S1
        state[1] = state[0]               # S2 = S1 at steady state
        state[2] = state[1] / (t_max_I * V_I * k_e) # I (plasma insulin concentration)

        # 2. Remote compartments (assume x = (k_b/k_a) * I at steady state)
        state[3] = (k_b1 / k_a1) * state[2] # x1
        state[4] = (k_b2 / k_a2) * state[2] # x2
        state[5] = (k_b3 / k_a3) * state[2] # x3

        # 3. Glucose
        g_init = params.get("G_init", 10.0)
        # Standardize G_init to mmol/L for internal ODE
        if g_init > 30: # Heuristic: if > 30, it's likely mg/dL
            g_init = g_init / 18.0182

        v_g = params.get("V_G", 0.16) * self.config.body_weight_kg
        state[6] = g_init * v_g  # Q1

        # 4. Q2 steady state: dQ2/dt = (k12 + x1)Q1 - (k12 + x2)Q2 = 0
        k_12 = params.get("k_12", 0.066)
        state[7] = state[6] * (k_12 + state[3]) / (k_12 + state[4])

        return state

    def reset(self, config: Optional[PatientConfig] = None) -> PatientState:
        if config:
            self.config = config
            self._p_array = pack_params(config.params, config.body_weight_kg)

        self._state_vec = self._initialize_state()
        self._time_min = 0
        self._f_sens = 1.0
        self._exercise_start_time = None
        self._exercise_end_time = None
        self._prev_exercise_active = False

        return self.state

    def step(self, t: int, insulin_rate_u_h: float, meal_carbs_g: float, exercise_active: bool) -> PatientState:
        self._time_min = t
        u_i_min = insulin_rate_u_h / 60.0

        # 1. Update exercise sensitivity
        self._update_exercise_sensitivity(exercise_active)

        # 2. Handle meal injection (D1 increment)
        if meal_carbs_g > 0:
            # Convert grams to mmol: carbs_mmol = meal_carbs * (1000.0 / 180.155)
            carbs_mmol = meal_carbs_g * (1000.0 / 180.155)
            self._state_vec[8] += carbs_mmol  # D1

        # 3. Step ODE and track rate (10 steps of 0.1 min to total 1 min)
        v_g = self.config.params.get("V_G", 0.16) * self.config.body_weight_kg
        q1_before = self._state_vec[6]

        for i in range(10):
            self._state_vec = hovorka_step(
                self._state_vec,
                float(t) + i * 0.1,
                u_i_min,
                self._f_sens,
                self._p_array,
                dt=0.1
            )

        q1_after = self._state_vec[6]
        dq1_dt = (q1_after - q1_before) / 1.0 # mmol/min
        self._last_glucose_rate = (dq1_dt / v_g) * 18.0182 # mg/dL/min

        return self.state

    def _update_exercise_sensitivity(self, active: bool) -> None:
        if active and not self._prev_exercise_active:
            self._exercise_start_time = self._time_min
            self._exercise_end_time = None
        elif not active and self._prev_exercise_active:
            self._exercise_end_time = self._time_min

        self._prev_exercise_active = active

        f_peak = self.config.params.get("F_peak", 1.35)
        k_rise = self.config.params.get("K_rise", 5.0)
        k_decay = self.config.params.get("K_decay", 0.01)

        if active and self._exercise_start_time is not None:
            t_rise = self._time_min - self._exercise_start_time
            self._f_sens = 1 + (f_peak - 1) * (1 - np.exp(-k_rise * t_rise))
        elif not active and self._exercise_end_time is not None:
            t_decay = self._time_min - self._exercise_end_time
            self._f_sens = 1 + (f_peak - 1) * np.exp(-k_decay * t_decay)
        else:
            self._f_sens = 1.0

    @property
    def time(self) -> int:
        """Current simulation time in minutes."""
        return self._time_min

    @property
    def basal_rate_uh(self) -> float:
        """Calculated weight-based basal rate (U/h)."""
        return self.config.body_weight_kg * 0.01

    @property
    def state(self) -> PatientState:
        v_g = self.config.params.get("V_G", 0.16) * self.config.body_weight_kg
        q1 = float(self._state_vec[6])
        glucose_mgdl = (q1 / v_g) * 18.0182

        # Use stored rate from last step, or 0 if just reset
        rate = getattr(self, "_last_glucose_rate", 0.0)

        return PatientState(
            time_min=self._time_min,
            glucose_mgdl=glucose_mgdl,
            glucose_rate_mgdl_min=rate,
            iob_u=float(self._state_vec[0] + self._state_vec[1]),
            cob_g=float((self._state_vec[8] + self._state_vec[9]) * (180.155 / 1000.0)),
            exercise_active=self._prev_exercise_active,
            compartments={
                "S1": float(self._state_vec[0]),
                "S2": float(self._state_vec[1]),
                "I": float(self._state_vec[2]),
                "x1": float(self._state_vec[3]),
                "x2": float(self._state_vec[4]),
                "x3": float(self._state_vec[5]),
                "Q1": float(self._state_vec[6]),
                "Q2": float(self._state_vec[7]),
                "D1": float(self._state_vec[8]),
                "D2": float(self._state_vec[9]),
            }
        )
