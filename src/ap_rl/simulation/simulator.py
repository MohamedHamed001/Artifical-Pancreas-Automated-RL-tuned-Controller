from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol
import numpy as np

from ap_rl.core.types import PatientConfig, PatientState, InsulinCommand, SafetyDecision
from ap_rl.simulation.patient import HovorkaPatientModel
from ap_rl.controllers.base import Controller
from ap_rl.controllers.safety import SafetySupervisor
from ap_rl.simulation.observations import ObservationBuilder
from ap_rl.rewards.standard import ClinicalZoneReward
from ap_rl.core.records import StepRecord, EpisodeRecord
from ap_rl.utils.insulin_calculator import InsulinCalculator


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    patient_config: PatientConfig
    controller: Controller
    duration_min: int = 1440
    dt_min: int = 5
    seed: int = 42
    target_glucose_mgdl: float = 120.0
    basal_rate_u_h: float = 1.0

    # Safety params
    min_glucose_mgdl: float = 70.0
    max_iob_factor: float = 3.0


class SimulationRunner:
    """
    Stabilized simulation runner.
    Orchestrates PatientModel, Controller, SafetyLayer, and Reward components.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Initialize Patient
        self.patient = HovorkaPatientModel(config.patient_config)

        # Initialize Controller
        self.controller = config.controller
        self.controller.reset()

        # Initialize Safety
        self.safety = SafetySupervisor(
            min_glucose_mgdl=config.min_glucose_mgdl,
            max_iob_factor=config.max_iob_factor,
            enable_low_suspend=True
        )

        # Initialize Reward
        self.reward_fn = ClinicalZoneReward(target_mgdl=config.target_glucose_mgdl)

        # Initialize Observation Builder
        self.obs_builder = ObservationBuilder(
            target_mgdl=config.target_glucose_mgdl,
            patient_weight=config.patient_config.body_weight_kg,
            dt_min=config.dt_min
        )

        # Initialize Bolus Manager
        self.insulin_calc = InsulinCalculator(
            patient_weight_kg=config.patient_config.body_weight_kg
        )

        self.current_time = 0

    def step(
        self,
        meal_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]] = None,
        exercise_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]] = None
    ) -> StepRecord:
        """
        Execute a single dt_min simulation step.
        """
        t = self.current_time
        current_state = self.patient.state

        # 1. Build Observation
        # Try to get gains from controller if it has them
        kp, ki, kd = 0.5, 0.001, 0.05
        if hasattr(self.controller, "pid"):
            inner = getattr(self.controller, "pid")
            kp, ki, kd = inner.Kp, inner.Ki, inner.Kd
        elif hasattr(self.controller, "Kp"):
            kp, ki, kd = self.controller.Kp, self.controller.Ki, self.controller.Kd

        is_exercising = self._get_scenario_exercise(t, exercise_data)
        obs = self.obs_builder.build(
            state=current_state,
            time_min=t,
            pid_gains=(kp, ki, kd),
            exercise_active=is_exercising
        )

        # 2. Controller Action (Basal portion)
        action = self.controller.get_action(obs, info={"time": t, "state": current_state})
        requested_u_h = float(action[0])

        # 3. Bolus Handling (Scenario-based)
        meal_g_min = self._get_scenario_meal(t, meal_data)
        bolus_rate_u_h = 0.0
        if meal_g_min > 0:
            self.insulin_calc.set_current_time(t)
            bolus_res = self.insulin_calc.deliver_bolus(
                meal_g_min,
                current_state.glucose_mgdl,
                self.config.target_glucose_mgdl
            )
            if bolus_res["delivered"]:
                # Deliver bolus as a 30-min drip (parity with legacy)
                bolus_rate_u_h = (bolus_res["immediate_dose"] / 30.0) * 60.0

        # Tail dose from previous boluses
        tail_rate_u_h = self.insulin_calc.drain_tail_dose()
        total_non_basal_rate = bolus_rate_u_h + tail_rate_u_h

        # 4. Safety Gating
        requested_command = InsulinCommand(
            time_min=t,
            basal_u_h=requested_u_h,
            bolus_u=0.0,
            total_u_h=requested_u_h + total_non_basal_rate
        )

        safety_decision = self.safety.evaluate(
            command=requested_command,
            state=current_state,
            basal_rate_uh=self.config.basal_rate_u_h
        )

        # 5. Physics Advancement (dt_min of 1-min steps)
        delivered_u_h = safety_decision.delivered.total_u_h

        for minute in range(self.config.dt_min):
            m_t = t + minute
            m_meal = self._get_scenario_meal(m_t, meal_data)
            m_ex = self._get_scenario_exercise(m_t, exercise_data)

            current_state = self.patient.step(
                t=m_t,
                insulin_rate_u_h=delivered_u_h,
                meal_carbs_g=m_meal,
                exercise_active=m_ex
            )

        # 6. Reward & Record
        reward = float(self.reward_fn.calculate(current_state))

        record = StepRecord(
            time=float(t),
            true_glucose=current_state.glucose_mgdl,
            observed_glucose=current_state.glucose_mgdl,
            requested_insulin=requested_command.total_u_h,
            delivered_insulin=delivered_u_h,
            basal=requested_u_h,
            bolus=total_non_basal_rate,
            iob=current_state.iob_u,
            cob=current_state.compartments.get("D1", 0.0) + current_state.compartments.get("D2", 0.0),
            Kp=kp, Ki=ki, Kd=kd,
            safety_events=safety_decision.active_constraints,
            reward=reward
        )

        self.current_time += self.config.dt_min

        # Update controller (optional learning)
        self.controller.update(reward=reward, done=False)

        return record

    def run(
        self,
        meal_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]] = None,
        exercise_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]] = None,
        scenario: Optional[Scenario] = None
    ) -> EpisodeRecord:
        """
        Execute a full simulation episode.
        """
        self.reset()
        if scenario is not None:
            meal_data = scenario.meal_data if scenario.meal_data is not None else scenario.meals
            exercise_data = scenario.exercise_data if scenario.exercise_data is not None else scenario.exercise

        steps = []
        total_reward = 0.0

        while self.current_time < self.config.duration_min:
            record = self.step(meal_data=meal_data, exercise_data=exercise_data)
            steps.append(record)
            total_reward += record.reward

        return EpisodeRecord(
            steps=steps,
            patient_id=self.config.patient_config.name,
            scenario_id=scenario.id if scenario else "simulation",
            total_reward=total_reward,
            controller_name=type(self.controller).__name__,
            meals=scenario.meals if scenario else [],
            exercise=scenario.exercise if scenario else []
        )

    def reset(self):
        """Reset runner state."""
        self.current_time = 0
        self.patient.reset()
        self.controller.reset()
        self.insulin_calc.reset()

    def _get_scenario_meal(self, t: int, meal_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]]) -> float:
        """Robust meal lookup supporting both numpy and list formats."""
        if meal_data is None: return 0.0
        if isinstance(meal_data, np.ndarray):
            if meal_data.size == 0: return 0.0
            idx = np.searchsorted(meal_data[:, 0], t, side='right') - 1
            if idx < 0: return 0.0
            return float(meal_data[idx, 1])
        else:
            for m in meal_data:
                if m.get("time") == t:
                    return float(m.get("value", m.get("carbs", 0.0)))
            return 0.0

    def _get_scenario_exercise(self, t: int, exercise_data: Optional[Union[np.ndarray, List[Dict[str, Any]]]]) -> bool:
        """Robust exercise lookup supporting both numpy and list formats."""
        if exercise_data is None: return False
        if isinstance(exercise_data, np.ndarray):
            if exercise_data.size == 0: return False
            idx = np.searchsorted(exercise_data[:, 0], t, side='right') - 1
            if idx < 0: return False
            return bool(exercise_data[idx, 1] > 0.5)
        else:
            for e in exercise_data:
                # Handle both "time" and "start" keys, and duration
                start = e.get("time", e.get("start", -1))
                duration = e.get("duration", 1)
                if start <= t < start + duration:
                    return True
            return False
