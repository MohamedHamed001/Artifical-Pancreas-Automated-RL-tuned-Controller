from __future__ import annotations
import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from ap_rl.core.types import PatientConfig, InsulinCommand
from ap_rl.controllers.pid_controller import PIDController
from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.utils.paths import data_dir
from ap_rl.evaluation.metrics import glucose_trajectory_summary


import gymnasium as gym
from gymnasium import spaces

class DiabetesPIDEnv(gym.Env):
    """
    Refactored Diabetes PID-tuning RL environment.
    Now utilizes SimulationRunner as the underlying engine.
    Inherits from gymnasium.Env for standard RL compatibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        patient_params: dict,
        patient_weight: float = 75.0,
        target_glucose: float = 120.0,
        data_root: Optional[str | os.PathLike] = None,
        seed: Optional[int] = None,
        observation_noise_std: float = 0.0,
        test_case_id: Optional[int] = None,
        carb_ratio_override: Optional[float] = None,
        isf_override: Optional[float] = None,
        demo_mode: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.patient_params = patient_params
        self.patient_weight = patient_weight
        self.target_glucose = target_glucose
        self.observation_noise_std = observation_noise_std
        self.demo_mode = demo_mode
        self._fixed_case_id = test_case_id
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Paths
        self.test_data_dir = os.fspath(data_root) if data_root else os.fspath(data_dir())

        # Load scenarios
        from ap_rl.utils.scenarios import ScenarioLoader
        self.loader = ScenarioLoader()
        self._load_random_test_case()

        # Initialize PID Controller
        self.pid = PIDController(
            target_mgdl=target_glucose,
            basal_u_h=self._estimate_basal(),
            Kp=0.5, Ki=0.1, Kd=0.01
        )

        # Initialize Simulation Runner
        patient_config = PatientConfig(
            name=f"Case_{getattr(self, 'current_case_id', 'unknown')}",
            params=self.patient_params,
            body_weight_kg=self.patient_weight
        )

        sim_config = SimulationConfig(
            patient_config=patient_config,
            controller=self.pid,
            duration_min=1440,
            dt_min=1, # Training environment uses 1-min steps
            seed=seed or 42,
            target_glucose_mgdl=target_glucose,
            basal_rate_u_h=self.pid.basal_u_h
        )

        self.runner = SimulationRunner(sim_config)

        # Env State
        self.max_episode_length = 1440
        self.done = False
        self.total_episode_reward = 0.0

        # History
        self.glucose_history: List[float] = []
        self.insulin_history: List[float] = []
        self.reward_history: List[float] = []

        # Define Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(3,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not getattr(self, "_skip_reload", False):
            self._load_random_test_case()
        self.runner.reset()
        self.done = False
        self.total_episode_reward = 0.0
        self.glucose_history = []
        self.insulin_history = []
        self.reward_history = []

        # Initial observation
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply (dKp, dKi, dKd) adjustment and advance 1 step.
        """
        # 1. Update PID Gains
        dkp, dki, dkd = action
        self.pid.Kp = float(np.clip(self.pid.Kp + dkp * 0.2, 0.01, 10.0))
        self.pid.Ki = float(np.clip(self.pid.Ki + dki * 0.1, 0.0, 1.0))
        self.pid.Kd = float(np.clip(self.pid.Kd + dkd * 0.1, 0.0, 1.0))

        # 2. Advance Simulation Runner by 1 min
        record = self.runner.step(
            meal_data=self.meal_data,
            exercise_data=self.exercise_data
        )

        # 3. Update Histories & Totals
        self.total_episode_reward += record.reward
        self.glucose_history.append(record.true_glucose)
        self.insulin_history.append(record.delivered_insulin)
        self.reward_history.append(record.reward)

        # 4. Termination Logic
        terminated = False
        truncated = False

        if self.runner.current_time >= self.max_episode_length:
            truncated = True

        if not self.demo_mode:
            if record.true_glucose < 40 or record.true_glucose > 500:
                terminated = True

        self.done = terminated or truncated

        info = {
            "glucose": record.true_glucose,
            "total_insulin": record.delivered_insulin,
            "basal_insulin": record.basal,
            "bolus_insulin": record.bolus,
            "iob": record.iob,
            "cob": record.cob,
            "Kp": self.pid.Kp,
            "Ki": self.pid.Ki,
            "Kd": self.pid.Kd,
            "safety_events": record.safety_events,
            "reward": record.reward
        }

        return self._get_obs(), record.reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation for agent."""
        # Use SimulationRunner's internal observation building logic
        t = self.runner.current_time
        state = self.runner.patient.state

        is_exercising = self.runner._get_scenario_exercise(t, self.exercise_data)

        obs = self.runner.obs_builder.build(
            state=state,
            time_min=t,
            pid_gains=(self.pid.Kp, self.pid.Ki, self.pid.Kd),
            exercise_active=is_exercising
        )

        # Add noise if configured
        if self.observation_noise_std > 0:
            obs[0] += self._rng.normal(0, self.observation_noise_std / 400.0)

        return obs

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set seed for environment."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def set_meal_schedule(self, meals: List[Dict[str, Any]]):
        """Set custom meal data."""
        self.meal_data = meals

    def set_exercise_schedule(self, exercise: List[Dict[str, Any]]):
        """Set custom exercise data."""
        self.exercise_data = exercise

    def get_statistics(self) -> Dict[str, Any]:
        """Compute episode statistics."""
        if not self.glucose_history:
            return {}

        stats = glucose_trajectory_summary(self.glucose_history)
        stats["total_insulin"] = float(sum(self.insulin_history))
        return stats

    @property
    def patient(self):
        """Compatibility property."""
        return self.runner.patient

    def _load_random_test_case(self):
        """Select a scenario."""
        if self._fixed_case_id is not None:
            case_id = self._fixed_case_id
        else:
            case_id = self._rng.integers(1, 11)

        self.current_case_id = case_id
        scenario = self.loader.load_case(case_id)
        self.meal_data = scenario.meal_data
        self.exercise_data = scenario.exercise_data

    def _estimate_basal(self) -> float:
        """Estimate basal insulin based on weight."""
        return (self.patient_weight * 0.01) # Rule of thumb: 0.01 U/kg/h
