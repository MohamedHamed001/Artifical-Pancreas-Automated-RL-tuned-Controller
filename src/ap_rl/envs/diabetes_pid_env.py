"""Diabetes PID-tuning RL environment.

Observation vector: 19-D (expanded from 16-D).
New features vs legacy:
  * ISF normalised, carb_ratio normalised, patient weight normalised
    — agent now knows *which patient* it is treating, enabling
    cross-profile generalisation without retraining.
  * Clinical zone-based reward (asymmetric hypo/hyper penalties that
    match real closed-loop system tuning guidelines).
  * Rate-of-change (dBGL/dt) penalty prevents "riding" rapid excursions.
  * IOB brake threshold is ISF-adaptive: fires earlier for sensitive patients.
  * Basal rate calculation uses the actual patient weight, not a hardcoded 75 kg.
"""

from __future__ import annotations

import glob
import math
import os
from typing import Optional

import numpy as np

from ap_rl.evaluation.metrics import glucose_trajectory_summary
from ap_rl.envs.hovorka_patient import HovorkaPatient
from ap_rl.utils.insulin_calculator import InsulinCalculator
from ap_rl.utils.meal_parser import MealParser
from ap_rl.utils.paths import data_dir
from ap_rl.utils.pid_controller import PID


class DiabetesPIDEnv:
    """Hovorka + PID + InsulinCalculator RL environment.

    Args:
        patient_params: Hovorka parameter dict (see
            ``configs/patient_default.yaml``).
        test_case_file: legacy compatibility argument; ignored if
            ``data_root`` resolves the scenario directory. Used for path
            inference when set to an absolute path.
        patient_weight: kg. Overwritten from the loaded scenario when
            ``TestCases.txt`` is available.
        target_glucose: mg/dL target setpoint.
        data_root: optional override for the TestData directory. Defaults
            to :func:`ap_rl.utils.paths.data_dir`.
        seed: optional integer seed for reproducible scenario selection.
        observation_noise_std: optional mg/dL std for additive Gaussian
            sensor noise applied to the **reported** BGL observation
            only. The underlying ODE remains noise-free.
        test_case_id: optional integer scenario index. When set, that
            scenario is always loaded (overrides ``seed``).
        carb_ratio_override / isf_override: optional clinical knobs
            forwarded to :class:`InsulinCalculator`.
    """

    def __init__(
        self,
        patient_params: dict,
        test_case_file: str | os.PathLike = "",
        patient_weight: float = 75.0,
        target_glucose: float = 120.0,
        data_root: Optional[str | os.PathLike] = None,
        seed: Optional[int] = None,
        observation_noise_std: float = 0.0,
        test_case_id: Optional[int] = None,
        carb_ratio_override: Optional[float] = None,
        isf_override: Optional[float] = None,
        demo_mode: bool = False,
    ) -> None:
        self.patient_params = patient_params
        self.patient_weight = patient_weight
        # If a non-default weight was passed, preserve it across episode resets
        # (don't let _get_body_weight_for_case overwrite it with TestCases.txt data).
        self._initial_patient_weight = patient_weight
        self._keep_patient_weight = (patient_weight != 75.0)
        # In demo/rollout mode, severe BGL excursions do NOT terminate the episode.
        # The simulation always runs to the full horizon so the app shows a complete
        # 24-hour graph.  The penalty reward is still applied for diagnostic value.
        self.demo_mode = demo_mode
        self.target_glucose = target_glucose
        self.patient: Optional[HovorkaPatient] = None

        self.iob_units = 0.0   # Real-time estimate of insulin units active in body
        self.max_episode_length = 1440
        self.bolus_duration = 30  # minutes for immediate drip portion
        
        if data_root is not None:
            self.test_data_dir = os.fspath(data_root)
        else:
            self.test_data_dir = os.fspath(data_dir())
        self.test_cases_file = os.path.join(self.test_data_dir, "TestCases.txt")

        meal_pattern = os.path.join(self.test_data_dir, "MealData_case*.data")
        exercise_pattern = os.path.join(self.test_data_dir, "ExerciseData_case*.data")
        self.meal_files = sorted(glob.glob(meal_pattern))
        self.exercise_files = sorted(glob.glob(exercise_pattern))

        self._rng = np.random.default_rng(seed)
        self._fixed_case_id = test_case_id
        self.observation_noise_std = float(observation_noise_std)

        self.parser = MealParser()

        self.meal_data: list[dict] = []
        self.exercise_data: list[dict] = []
        self._load_random_test_case()

        self.insulin_calc = InsulinCalculator(
            patient_weight_kg=self.patient_weight,
            carb_ratio=carb_ratio_override,
            isf=isf_override,
        )
        self._carb_ratio_override = carb_ratio_override
        self._isf_override = isf_override

        self.pid = PID(P=0.5, I=0.1, D=0.01)
        self.pid.SetPoint = target_glucose
        self.pid.setSampleTime(1.0)
        self.pid.setWindup(50.0)

        self.max_episode_length = 1440
        self.current_step = 0
        self.done = False

        self.observation_space = 19   # expanded: +ISF, +carb_ratio, +weight
        self.action_space = 3
        self.action_bound = 0.1

        self.glucose_history: list[float] = []
        self.insulin_history: list[float] = []
        self.pid_history: dict[str, list[float]] = {"Kp": [], "Ki": [], "Kd": []}
        self.reward_history: list[float] = []
        self.bolus_history: list[dict] = []

        self.previous_glucose = target_glucose
        self.prev2_glucose = target_glucose       # for 2-min rate
        self.time_since_last_meal = 1440
        self.time_since_last_insulin = 1440
        self.total_episode_reward = 0.0

        self.bolus_duration = 15
        self.bolus_remaining = 0.0
        self.bolus_rate = 0.0

    def seed(self, seed: Optional[int]) -> None:
        """Re-seed the scenario-selection RNG."""
        self._rng = np.random.default_rng(seed)

    def _load_random_test_case(self) -> None:
        if not (self.meal_files and self.exercise_files):
            self.meal_data = []
            self.exercise_data = []
            return

        case_numbers: list[int] = []
        for meal_file in self.meal_files:
            basename = os.path.basename(meal_file)
            num = basename.replace("MealData_case", "").replace(".data", "")
            case_numbers.append(int(num))

        if self._fixed_case_id is not None:
            selected_case = int(self._fixed_case_id)
        else:
            selected_case = int(self._rng.choice(case_numbers))

        meal_file = os.path.join(self.test_data_dir, f"MealData_case{selected_case}.data")
        exercise_file = os.path.join(
            self.test_data_dir, f"ExerciseData_case{selected_case}.data"
        )

        if not getattr(self, "_keep_patient_weight", False):
            self.patient_weight = self._get_body_weight_for_case(selected_case)
        self.meal_data = self._parse_data_file(meal_file, "meal")
        self.exercise_data = self._parse_data_file(exercise_file, "exercise")
        self.current_case_id = selected_case

    def _parse_data_file(self, file_path: str, data_type: str) -> list[dict]:
        data: list[dict] = []
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    time_val = float(parts[0])
                    if data_type == "meal":
                        carbs_val = float(parts[1])
                        if carbs_val > 0:
                            data.append({"time": time_val, "carbs": carbs_val})
                    else:
                        active_val = int(float(parts[1]))
                        data.append({"time": time_val, "active": active_val})
        except FileNotFoundError:
            print(f"Warning: Data file {file_path} not found")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Error parsing {file_path}: {exc}")
        return data

    def _get_body_weight_for_case(self, case_number: int) -> float:
        try:
            with open(self.test_cases_file, "r") as f:
                lines = f.readlines()
            current_case: Optional[int] = None
            for line in lines:
                line = line.strip()
                if line.startswith(f"Test Case [{case_number}]"):
                    current_case = case_number
                    continue
                if current_case == case_number and line.startswith("Body Weight:"):
                    weight_str = line.split(":")[1].strip().replace(" kg", "")
                    return float(weight_str)
                if current_case == case_number and line.startswith("Test Case ["):
                    break
        except Exception:
            pass
        return 75.0

    def _save_test_case_to_files(
        self, meal_file_path: str, exercise_file_path: str
    ) -> None:
        with open(meal_file_path, "w") as f:
            f.write("# Time(min) Carbs(g)\n")
            for meal in self.meal_data:
                f.write(f"{meal['time']} {meal['carbs']}\n")
                f.write(f"{meal['time'] + 1} 0.0\n")
        with open(exercise_file_path, "w") as f:
            f.write("# Time(min) Active(0/1)\n")
            for exercise in self.exercise_data:
                f.write(f"{exercise['time']} {exercise['active']}\n")

    def set_meal_schedule(self, meals: list[dict]) -> None:
        """Inject a deterministic meal schedule.

        Each entry should be ``{"time": minutes, "carbs": grams}``. Must be
        called **before** :meth:`reset` to take effect for the next episode.
        """
        self.meal_data = list(meals)

    def set_exercise_schedule(self, events: list[dict]) -> None:
        """Inject a deterministic exercise schedule (``time``, ``active``)."""
        self.exercise_data = list(events)

    def reset(self) -> np.ndarray:
        """Start a new episode, returning the initial observation."""
        if self._fixed_case_id is None and not getattr(self, "_skip_reload", False):
            self._load_random_test_case()

        self.patient = HovorkaPatient(patient_params=self.patient_params)

        os.makedirs("temp_data", exist_ok=True)
        meal_temp = "temp_data/meal_temp.data"
        exercise_temp = "temp_data/exercise_temp.data"
        self._save_test_case_to_files(meal_temp, exercise_temp)

        self.patient.load_meal_data(meal_temp)
        self.patient.load_exercise_data(exercise_temp)

        self.pid.clear()
        self.pid.SetPoint = self.target_glucose
        self.pid.Kp = 0.5
        self.pid.Ki = 0.1
        self.pid.Kd = 0.01
        self.pid.setWindup(50.0)

        self.insulin_calc = InsulinCalculator(
            patient_weight_kg=self.patient_weight,
            carb_ratio=self._carb_ratio_override,
            isf=self._isf_override,
        )

        self.current_step = 0
        self.done = False
        self.previous_glucose = self.target_glucose
        self.prev2_glucose = self.target_glucose
        self.time_since_last_meal = 1440
        self.time_since_last_insulin = 1440
        self.total_episode_reward = 0.0

        self.bolus_remaining = 0.0
        self.bolus_rate = 0.0
        self.iob_units = 0.0

        self.glucose_history = []
        self.insulin_history = []
        self.pid_history = {"Kp": [], "Ki": [], "Kd": []}
        self.reward_history = []
        self.bolus_history = []

        return self._get_state()

    def _observe_glucose(self) -> float:
        """Return the (possibly noisy) reported BGL in mg/dL.

        The underlying ODE state is not modified - this only perturbs the
        observation handed to the controller.
        """
        bgl_truth = self.patient.G * 18.0182  # type: ignore[union-attr]
        if self.observation_noise_std > 0.0:
            return float(bgl_truth + self._rng.normal(0.0, self.observation_noise_std))
        return float(bgl_truth)

    def _get_state(self) -> np.ndarray:
        """Build 19-D observation vector.

        Features 1–13: original legacy features (glucose, rate, error,
        PID terms + gains, timing, exercise, circadian).
        Features 14–16: anti-rage-bolus additions (rate-2, tail-bolus, IOB).
        Features 17–19 (NEW): patient identity — ISF, carb_ratio, weight.
            These tell the agent *how sensitive* the patient is so it can
            calibrate its aggressiveness without seeing the patient profile
            directly.  Normalised to roughly [0, 1]:
            ISF   / 100  (typical range 20–80 mg/dL per U)
            CR    / 20   (typical range 8–20 g/U)
            BW    / 120  (typical range 50–110 kg)
        """
        current_glucose = self._observe_glucose()
        glucose_rate   = current_glucose - self.previous_glucose          # 1-min
        glucose_rate_2 = (current_glucose - self.prev2_glucose) / 2.0    # 2-min avg
        error = self.target_glucose - current_glucose

        hour_of_day = (self.patient.time % 1440) / 60.0  # type: ignore[union-attr]
        time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

        exercise_status = self.patient._get_exercise_status(self.patient.time)  # type: ignore[union-attr]

        # Bolus remaining normalised (10 U is a large bolus)
        bolus_remaining_norm = min(self.insulin_calc.pending_tail_dose / 10.0, 1.0)

        # IOB: Real Units On Board (normalised by 10 U)
        iob_norm = float(np.clip(self.iob_units / 10.0, 0.0, 1.5))

        # Patient identity features (normalised)
        isf_norm    = float(np.clip(self.insulin_calc.isf / 100.0, 0.0, 1.5))
        cr_norm     = float(np.clip(self.insulin_calc.carb_ratio / 20.0, 0.0, 1.5))
        weight_norm = float(np.clip(self.patient_weight / 120.0, 0.0, 1.5))

        state = np.array(
            [
                current_glucose / 400.0,       # 1
                glucose_rate / 100.0,           # 2
                error / 200.0,                  # 3
                self.pid.ITerm / 100.0,         # 4
                self.pid.DTerm / 10.0,          # 5
                self.pid.Kp / 10.0,             # 6 (Normalised 0.0-1.0)
                self.pid.Ki,                    # 7
                self.pid.Kd,                    # 8
                min(self.time_since_last_meal / 240.0, 1.0),      # 9
                min(self.time_since_last_insulin / 60.0, 1.0),    # 10
                exercise_status,                # 11
                time_sin,                       # 12
                time_cos,                       # 13
                glucose_rate_2 / 100.0,         # 14
                bolus_remaining_norm,           # 15
                iob_norm,                       # 16
                # --- NEW patient-identity features ---
                isf_norm,                       # 17
                cr_norm,                        # 18
                weight_norm,                    # 19
            ],
            dtype=np.float32,
        )
        return state

    def _calculate_reward(self, glucose_mgdl: float, glucose_diff: float, insulin_delivered: float) -> float:
        """Clinical zone-model reward function.

        Based on validated closed-loop AP tuning guidelines (Kovatchev et al.,
        Diabetes Care 2009).  Zones and penalty asymmetry match clinical intent:

        Zone A (70–180): primary comfort zone — Gaussian pull toward 100 mg/dL.
        Zone B (54–70 / 180–250): early warning — linear penalties.
        Zone C (<54 / >250): danger — quadratic+ penalties.
        Zone D (<40): severe — episode termination.

        Additional shaping:
        - Rate-of-change penalty: prevents the agent from riding rapid excursions.
        - Patient-adaptive IOB brake: threshold scales with the patient's ISF
          so insulin-sensitive patients get protection at lower Kp values.
        - Recovery bonus and stability bonus preserved from prior version.
        """
        # ----------------------------------------------------------------
        # Termination safety cutoffs
        # ----------------------------------------------------------------
        if glucose_mgdl < 40:
            if not self.demo_mode:
                self.done = True
            return -10000.0   # Catastrophic: must never be profitable to crash
        elif glucose_mgdl > 500:
            if not self.demo_mode:
                self.done = True
            return -3000.0

        reward = 0.0

        # ----------------------------------------------------------------
        # 1. ZONE MODEL (primary signal)
        # Asymmetry: hypo is ~3x worse than equivalent hyper (clinical consensus).
        # ----------------------------------------------------------------
        if 70.0 <= glucose_mgdl <= 180.0:
            # TIR zone — Gaussian centred on 100 mg/dL (not 120) to pull
            # the agent toward the lower-normal sweet spot while staying safe.
            distance = abs(glucose_mgdl - 100.0)
            reward += 50.0 * math.exp(-(distance ** 2) / (2 * 35.0 ** 2))

        elif 54.0 <= glucose_mgdl < 70.0:
            # Zone B hypo: linear penalty, starts mild
            undershoot = 70.0 - glucose_mgdl
            reward -= undershoot * 4.0       # up to -64 at BGL=54

        elif glucose_mgdl < 54.0:
            # Zone C/D hypo: quadratic — gets dangerous fast
            undershoot = 70.0 - glucose_mgdl
            reward -= (undershoot ** 2) * 1.5

        elif 180.0 < glucose_mgdl <= 250.0:
            # Zone B hyper: linear penalty
            overshoot = glucose_mgdl - 180.0
            reward -= overshoot * 1.2        # up to -84 at BGL=250

        elif glucose_mgdl > 250.0:
            # Zone C hyper: quadratic
            overshoot = glucose_mgdl - 180.0
            reward -= (overshoot ** 1.8) * 0.15

        # ----------------------------------------------------------------
        # 2. RATE-OF-CHANGE PENALTY
        # A healthy pancreas never lets BGL move faster than ±2 mg/dL/min.
        # Penalise rapid excursions in either direction.
        # ----------------------------------------------------------------
        abs_rate = abs(glucose_diff)
        if abs_rate > 3.0:
            reward -= (abs_rate - 3.0) ** 2 * 2.0
        elif abs_rate > 2.0:
            reward -= (abs_rate - 2.0) * 1.5

        # ----------------------------------------------------------------
        # 3. PROACTIVE IOB BRAKE (Anti-Rage Bolus)
        # ----------------------------------------------------------------
        iob = self.iob_units
        isf = self.insulin_calc.isf
        kp_brake_threshold = max(0.8, isf / 40.0)
        
        # Penalise high gain if we already have significant insulin active.
        # This now triggers even during the rise if IOB is excessive,
        # preventing the 'integrator windup' of the Kp gain.
        if iob > 1.5 and self.pid.Kp > kp_brake_threshold:
            excess_kp = self.pid.Kp - kp_brake_threshold
            reward -= (excess_kp * iob) * 20.0  # Increased penalty
        
        # Additional brake: if BGL is already falling and we still have high gain
        if glucose_diff < -0.2 and iob > 0.5 and self.pid.Kp > kp_brake_threshold:
            reward -= (self.pid.Kp - kp_brake_threshold) * 50.0

        # ----------------------------------------------------------------
        # 4. RECOVERY & STABILITY
        # ----------------------------------------------------------------
        # Reward controlled descent from hyperglycemia (not a free-fall crash)
        if glucose_mgdl > 200 and -4.0 < glucose_diff < -0.5:
            reward += 8.0

        # Note: 'Proactive anticipation bonus' removed. It was a perverse 
        # incentive that encouraged the agent to ramp Kp regardless of risk.

        # ----------------------------------------------------------------
        # 5. STABILITY BONUS
        # ----------------------------------------------------------------
        if abs(glucose_diff) <= 1.5:
            reward += 8.0   # increased: strongly reward steady-state behaviour
        elif abs(glucose_diff) <= 3.0:
            reward += 3.0

        return float(reward)

    def _safety_clamp(self, basal_rate: float, glucose: float, glucose_rate: float) -> float:
        """Rule-based safety layer: suspend or reduce basal on hypoglycemia risk.

        This is a hard constraint applied on top of the RL/PID output and
        cannot be over-ridden by the agent.
        """
        # Hard suspend: confirmed or imminent severe hypo
        if glucose < 70:
            return 0.0
        if glucose < 80 and glucose_rate < -1.5:
            return 0.0          # predictive suspension (falling fast)
        # Significant reduction during descent toward hypo
        if glucose < 90 and glucose_rate < -1.0:
            return basal_rate * 0.3
        # Mild reduction if drifting down from marginal range
        if glucose < 100 and glucose_rate < -0.5:
            return basal_rate * 0.6
        return basal_rate

    def _handle_meal_bolus(self) -> float:
        current_meal = self.patient._get_meal_intake(self.patient.time)  # type: ignore[union-attr]
        current_glucose = self.patient.G * 18.0182  # type: ignore[union-attr]

        if current_meal > 0:
            self.insulin_calc.set_current_time(self.patient.time)  # type: ignore[union-attr]
            bolus_result = self.insulin_calc.deliver_bolus(
                carbs_grams=current_meal,
                current_glucose_mgdl=current_glucose,
                target_glucose_mgdl=self.target_glucose,
            )
            if bolus_result["delivered"]:
                # Use IMMEDIATE portion only; tail drips via drain_tail_dose()
                immediate = bolus_result["immediate_dose"]
                self.bolus_remaining = immediate
                self.bolus_rate = (immediate / self.bolus_duration) * 60
                self.time_since_last_meal = 0
                self.time_since_last_insulin = 0

                self.bolus_history.append(
                    {
                        "time": self.patient.time,  # type: ignore[union-attr]
                        "carbs": current_meal,
                        "bolus_dose": bolus_result["bolus_dose"],
                        "correction_dose": bolus_result["correction_dose"],
                        "total_dose": bolus_result["total_dose"],
                        "immediate_dose": bolus_result["immediate_dose"],
                        "tail_dose": bolus_result["tail_dose"],
                    }
                )
                return self.bolus_rate
        return 0.0

    def _get_bolus_insulin(self) -> float:
        """Return this-minute bolus delivery (U/h): immediate drip + tail drip."""
        immediate_rate = 0.0
        if self.bolus_remaining > 0:
            bolus_this_minute = min(self.bolus_remaining, self.bolus_rate / 60)
            self.bolus_remaining -= bolus_this_minute
            immediate_rate = bolus_this_minute * 60
        # Tail drip from InsulinCalculator (already returns U/h)
        tail_rate = self.insulin_calc.drain_tail_dose()
        return immediate_rate + tail_rate

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply ``(dKp, dKi, dKd)``, advance the simulator, return ``(obs, r, done, info)``."""
        delta_kp, delta_ki, delta_kd = action

        # Scale the deltas to prevent erratic jumping. 
        # Reducing Kp scaling from 0.5 to 0.2 to prevent "rage bolusing" oscillations
        delta_kp *= 0.2
        delta_ki *= 0.1
        delta_kd *= 0.1

        self.pid.Kp = float(np.clip(self.pid.Kp + delta_kp, 0.01, 10.0))
        self.pid.Ki = float(np.clip(self.pid.Ki + delta_ki, 0.0, 1.0))
        self.pid.Kd = float(np.clip(self.pid.Kd + delta_kd, 0.0, 1.0))

        current_glucose = self.patient.G * 18.0182  # type: ignore[union-attr]
        glucose_rate    = current_glucose - self.previous_glucose

        _ = self._handle_meal_bolus()
        bolus_rate = self._get_bolus_insulin()

        if bolus_rate > 0:
            basal_rate = 0.5
        else:
            # --- Asymmetric PID anti-windup: freeze ITerm during descent ---
            if current_glucose < 80 and glucose_rate < 0:
                # Prevent integral from accumulating insulin during hypoglycemia
                saved_iterm = self.pid.ITerm
                self.pid.SetPoint = self.target_glucose
                self.pid.update(current_glucose)
                self.pid.ITerm = saved_iterm   # restore — no windup during hypo
            else:
                self.pid.SetPoint = self.target_glucose
                self.pid.update(current_glucose)

            # Use actual patient weight, not a hardcoded 75 kg.
            # TDI ≈ 0.55 U/kg/day; basal ≈ 50% of TDI split across 24 h.
            estimated_tdd = self.patient_weight * 0.55
            base_basal_rate = estimated_tdd * 0.5 / 24.0
            pid_adjustment = -self.pid.output * 0.01
            basal_rate = float(np.clip(base_basal_rate + pid_adjustment, 0.0, 10.0))

        # --- Safety clamp (hard constraint, cannot be over-ridden by RL) ---
        basal_rate = self._safety_clamp(basal_rate, current_glucose, glucose_rate)

        total_insulin_rate = basal_rate + bolus_rate

        # Advance glucose history (keep prev2 for 2-min rate in _get_state)
        self.prev2_glucose = self.previous_glucose
        self.previous_glucose = current_glucose
        new_glucose = self.patient.step(total_insulin_rate)  # type: ignore[union-attr]

        # Update real-time IOB estimate (Units On Board)
        # 45-min half-life => ~0.985 decay per minute.
        self.iob_units = (self.iob_units * 0.985) + (total_insulin_rate / 60.0)

        self.time_since_last_meal += 1
        self.time_since_last_insulin += 1
        if total_insulin_rate > 0:
            self.time_since_last_insulin = 0

        reward = self._calculate_reward(new_glucose, new_glucose - current_glucose, total_insulin_rate)
        self.total_episode_reward += reward

        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            self.done = True

        self.glucose_history.append(new_glucose)
        self.insulin_history.append(total_insulin_rate)
        self.pid_history["Kp"].append(self.pid.Kp)
        self.pid_history["Ki"].append(self.pid.Ki)
        self.pid_history["Kd"].append(self.pid.Kd)
        self.reward_history.append(reward)

        next_state = self._get_state()
        info = {
            "glucose": new_glucose,
            "basal_insulin": basal_rate,
            "bolus_insulin": bolus_rate,
            "total_insulin": total_insulin_rate,
            "Kp": self.pid.Kp,
            "Ki": self.pid.Ki,
            "Kd": self.pid.Kd,
            "episode_reward": self.total_episode_reward,
            "step": self.current_step,
        }
        return next_state, float(reward), bool(self.done), info

    def render(self, mode: str = "human") -> None:
        if not self.glucose_history:
            print("No data to render yet.")
            return
        if mode == "human":
            print(
                f"Step {self.current_step}: BGL={self.glucose_history[-1]:.1f} mg/dL, "
                f"Insulin={self.insulin_history[-1]:.2f} U/h, "
                f"PID=[{self.pid.Kp:.3f}, {self.pid.Ki:.3f}, {self.pid.Kd:.3f}], "
                f"Reward={self.reward_history[-1]:.2f}"
            )

    def get_statistics(self) -> dict:
        if not self.glucose_history:
            return {}
        glucose_array = np.array(self.glucose_history)
        out = glucose_trajectory_summary(glucose_array)
        out.update(
            {
                "total_episode_reward": self.total_episode_reward,
                "mean_insulin": float(np.mean(self.insulin_history)),
                "total_insulin": float(np.sum(self.insulin_history) / 60),
                "final_kp": self.pid.Kp,
                "final_ki": self.pid.Ki,
                "final_kd": self.pid.Kd,
                "num_boluses": len(self.bolus_history),
            }
        )
        return out


import atexit


def _cleanup_temp_files() -> None:  # pragma: no cover - best-effort
    import shutil

    if os.path.exists("temp_data"):
        try:
            shutil.rmtree("temp_data")
        except OSError:
            pass


atexit.register(_cleanup_temp_files)
