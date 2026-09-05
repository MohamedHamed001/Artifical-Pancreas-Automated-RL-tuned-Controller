import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ap_rl.core.types import PatientConfig
from ap_rl.simulation.legacy_patient import HovorkaPatient
from ap_rl.controllers.safety import SafetySupervisor
from ap_rl.core.records import StepRecord, EpisodeRecord
from ap_rl.rewards.clinical import ClinicalReward
from ap_rl.evaluation.metrics import glucose_trajectory_summary


class GlucoseControlEnv(gym.Env):
    """
    A research-ready Gymnasium environment for direct insulin control.
    Uses the unified Hovorka physics core and SafetySupervisor.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        patient_config: PatientConfig,
        dt_min: int = 5,
        max_duration_min: int = 1440,
        target_glucose: float = 120.0,
        basal_rate: float = 1.0,
        reward_fn: Optional[Any] = None,
    ):
        super().__init__()
        self.patient_config = patient_config
        self.patient_params = patient_config.params
        self.patient_weight = patient_config.body_weight_kg
        self.dt_min = dt_min
        self.max_duration_min = max_duration_min
        self.target_glucose = target_glucose
        self.basal_rate = basal_rate
        self.reward_fn = reward_fn

        # Action space: Direct insulin command (U/h)
        # Bounded between 0 and 10 U/h (adjustable)
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)

        # Observation space: [Glucose, Glucose_Velocity, IOB, COB, Time_Sin, Time_Cos]
        # We use a broad observation space.
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, -1, -1], dtype=np.float32),
            high=np.array([1000, 10, 50, 200, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.patient = HovorkaPatient(self.patient_params)
        self.safety = SafetySupervisor(
            min_glucose_mgdl=70.0,
            max_iob_factor=3.0
        )

        self.current_time = 0
        self.last_glucose = 0.0
        self.steps: List[StepRecord] = []
        self.glucose_history: List[float] = []
        self.insulin_history: List[float] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize Patient
        self.patient.reset()
        if options and "meal_data" in options:
            self.patient.set_meal_data(options["meal_data"])

        self.current_time = 0
        self.last_glucose = self.patient.G * 18.0182
        self.steps = []
        self.glucose_history = []
        self.insulin_history = []

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        requested_insulin = float(action[0])

        # 1. Apply Safety Supervisor
        from ap_rl.core.types import PatientState, InsulinCommand

        current_glucose = self.patient.G * 18.0182
        glucose_velocity = (current_glucose - self.last_glucose) / self.dt_min
        current_iob = (self.patient.S1 + self.patient.S2)
        current_cob = self.patient.D1 + self.patient.D2

        state = PatientState(
            time_min=self.current_time,
            glucose_mgdl=current_glucose,
            glucose_rate_mgdl_min=glucose_velocity,
            iob_u=current_iob,
            cob_g=current_cob,
            exercise_active=False, # Simplification for now
            compartments={} # Legacy patient doesn't expose internal compartments easily
        )

        cmd = InsulinCommand(
            time_min=self.current_time,
            basal_u_h=requested_insulin,
            total_u_h=requested_insulin
        )

        decision = self.safety.evaluate(
            command=cmd,
            state=state,
            basal_rate_uh=self.basal_rate
        )
        delivered_insulin = decision.delivered.total_u_h
        safety_events = decision.active_constraints

        self.last_glucose = current_glucose

        # 2. Advance simulation
        for _ in range(self.dt_min):
            p_res = self.patient.step(delivered_insulin)
            self.current_time += 1

        # 3. Record
        record = StepRecord(
            time=float(self.current_time),
            true_glucose=p_res["glucose"],
            observed_glucose=p_res["glucose"],
            requested_insulin=requested_insulin,
            delivered_insulin=delivered_insulin,
            basal=self.basal_rate,
            bolus=0.0,
            iob=current_iob,
            cob=self.patient.D1 + self.patient.D2,
            safety_events=safety_events
        )
        self.steps.append(record)
        self.glucose_history.append(record.true_glucose)
        self.insulin_history.append(record.delivered_insulin)

        # 4. Calculate Reward
        reward = self._calculate_reward(p_res["glucose"], delivered_insulin, safety_events)

        # 5. Check termination/truncation
        terminated = p_res["glucose"] < 40 or p_res["glucose"] > 500
        truncated = self.current_time >= self.max_duration_min

        obs = self._get_obs()
        info = self._get_info()
        info["safety_events"] = safety_events

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        current_glucose = self.patient.G * 18.0182
        glucose_velocity = (current_glucose - self.last_glucose) / self.dt_min
        iob = (self.patient.S1 + self.patient.S2)
        cob = self.patient.D1 + self.patient.D2

        time_hours = (self.current_time / 60.0) % 24
        time_sin = np.sin(2 * np.pi * time_hours / 24.0)
        time_cos = np.cos(2 * np.pi * time_hours / 24.0)

        return np.array([
            current_glucose,
            glucose_velocity,
            iob,
            cob,
            time_sin,
            time_cos
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "glucose": self.patient.G * 18.0182,
            "iob": self.patient.S1 + self.patient.S2,
            "cob": self.patient.D1 + self.patient.D2,
            "time": self.current_time
        }

    def _calculate_reward(self, glucose: float, insulin: float, safety_events: list = []) -> float:
        """
        Calculates the clinical reward using the dedicated ClinicalReward module.
        """
        if self.reward_fn is not None:
            return float(self.reward_fn(glucose, insulin, safety_events))

        reward_engine = ClinicalReward(target_glucose=self.target_glucose)
        return reward_engine.compute_reward(
            glucose_mgdl=glucose,
            insulin_delivered=insulin,
            safety_events=safety_events
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Compute episode statistics."""
        if not self.glucose_history:
            return {}

        stats = glucose_trajectory_summary(self.glucose_history)
        stats["total_insulin"] = float(sum(self.insulin_history))
        return stats
