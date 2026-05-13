"""Diabetes PID-tuning RL environment.

Behaviour preserved verbatim from the legacy
``RL_Diabetes_Controller/envs/diabetes_pid_env.py`` implementation:

* 13-D observation vector.
* 3-D action (delta_Kp, delta_Ki, delta_Kd) bounded to ``+-0.1``.
* Identical reward shaping (target band, stability bonus, etc.).
* PID + InsulinCalculator for meal bolus + correction, basal via PID
  output negated and scaled.

New (non-breaking) features added by the refactor:

* Optional ``seed`` argument makes ``_load_random_test_case`` deterministic.
* Optional ``observation_noise_std`` parameter adds Gaussian noise to the
  reported BGL observation only (does **not** corrupt the true ODE state),
  matching the "unstable" profile design.
* Optional explicit ``test_case_id`` so the demo can pin a scenario.
* ``run_episode`` helper-friendly: state shape and step semantics unchanged.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

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
    ) -> None:
        self.patient_params = patient_params
        self.patient_weight = patient_weight
        self.target_glucose = target_glucose
        self.patient: Optional[HovorkaPatient] = None

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

        self.observation_space = 13
        self.action_space = 3
        self.action_bound = 0.1

        self.glucose_history: list[float] = []
        self.insulin_history: list[float] = []
        self.pid_history: dict[str, list[float]] = {"Kp": [], "Ki": [], "Kd": []}
        self.reward_history: list[float] = []
        self.bolus_history: list[dict] = []

        self.previous_glucose = target_glucose
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
        self.time_since_last_meal = 1440
        self.time_since_last_insulin = 1440
        self.total_episode_reward = 0.0

        self.bolus_remaining = 0.0
        self.bolus_rate = 0.0

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
        current_glucose = self._observe_glucose()
        glucose_rate = current_glucose - self.previous_glucose
        error = self.target_glucose - current_glucose

        hour_of_day = (self.patient.time % 1440) / 60.0  # type: ignore[union-attr]
        time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

        exercise_status = self.patient._get_exercise_status(self.patient.time)  # type: ignore[union-attr]

        state = np.array(
            [
                current_glucose / 400.0,
                glucose_rate / 100.0,
                error / 200.0,
                self.pid.ITerm / 100.0,
                self.pid.DTerm / 10.0,
                self.pid.Kp,
                self.pid.Ki,
                self.pid.Kd,
                min(self.time_since_last_meal / 240.0, 1.0),
                min(self.time_since_last_insulin / 60.0, 1.0),
                exercise_status,
                time_sin,
                time_cos,
            ],
            dtype=np.float32,
        )
        return state

    def _calculate_reward(self, glucose_mgdl: float, insulin_delivered: float) -> float:
        """Original safety-first reward (unchanged)."""
        reward = 0.0

        if glucose_mgdl < 40:
            reward = -500
            self.done = True
        elif glucose_mgdl > 300:
            reward = -500
            self.done = True
        elif glucose_mgdl < 50:
            reward = -200 - (50 - glucose_mgdl) * 10
        elif glucose_mgdl > 250:
            reward = -200 - (glucose_mgdl - 250) * 2
        elif 80 <= glucose_mgdl <= 140:
            if 90 <= glucose_mgdl <= 120:
                reward = 20
            else:
                reward = 15
        elif 70 <= glucose_mgdl < 80 or 140 < glucose_mgdl <= 180:
            reward = 5
        elif glucose_mgdl < 70:
            reward = -15 - (70 - glucose_mgdl) * 0.8
        elif glucose_mgdl > 180:
            reward = -10 - (glucose_mgdl - 180) * 0.15

        glucose_rate = glucose_mgdl - self.previous_glucose
        if abs(glucose_rate) <= 5:
            reward += 3
        elif abs(glucose_rate) <= 10:
            reward += 1
        elif abs(glucose_rate) > 20:
            reward -= abs(glucose_rate) * 0.2

        if hasattr(self, "consecutive_in_range"):
            if 70 <= glucose_mgdl <= 180:
                self.consecutive_in_range += 1
                if self.consecutive_in_range >= 60:
                    reward += 5
            else:
                self.consecutive_in_range = 0
        else:
            self.consecutive_in_range = 1 if 70 <= glucose_mgdl <= 180 else 0

        if insulin_delivered > 10:
            reward -= (insulin_delivered - 10) * 0.8

        kp_change = abs(self.pid.Kp - getattr(self, "prev_kp", self.pid.Kp))
        ki_change = abs(self.pid.Ki - getattr(self, "prev_ki", self.pid.Ki))
        kd_change = abs(self.pid.Kd - getattr(self, "prev_kd", self.pid.Kd))
        if kp_change + ki_change + kd_change < 0.1:
            reward += 1

        self.prev_kp = self.pid.Kp
        self.prev_ki = self.pid.Ki
        self.prev_kd = self.pid.Kd

        return reward

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
                self.bolus_remaining = bolus_result["total_dose"]
                self.bolus_rate = self.bolus_remaining / self.bolus_duration * 60
                self.time_since_last_meal = 0
                self.time_since_last_insulin = 0

                self.bolus_history.append(
                    {
                        "time": self.patient.time,  # type: ignore[union-attr]
                        "carbs": current_meal,
                        "bolus_dose": bolus_result["bolus_dose"],
                        "correction_dose": bolus_result["correction_dose"],
                        "total_dose": bolus_result["total_dose"],
                    }
                )
                return self.bolus_rate
        return 0.0

    def _get_bolus_insulin(self) -> float:
        if self.bolus_remaining > 0:
            bolus_this_minute = min(self.bolus_remaining, self.bolus_rate / 60)
            self.bolus_remaining -= bolus_this_minute
            return bolus_this_minute * 60
        return 0.0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply ``(dKp, dKi, dKd)``, advance the simulator, return ``(obs, r, done, info)``."""
        delta_kp, delta_ki, delta_kd = action

        self.pid.Kp = float(np.clip(self.pid.Kp + delta_kp, 0.01, 2.0))
        self.pid.Ki = float(np.clip(self.pid.Ki + delta_ki, 0.0, 0.01))
        self.pid.Kd = float(np.clip(self.pid.Kd + delta_kd, 0.0, 0.1))

        current_glucose = self.patient.G * 18.0182  # type: ignore[union-attr]

        _ = self._handle_meal_bolus()
        bolus_rate = self._get_bolus_insulin()

        if bolus_rate > 0:
            basal_rate = 0.5
        else:
            self.pid.SetPoint = self.target_glucose
            self.pid.update(current_glucose)

            patient_weight = 75
            estimated_tdd = patient_weight * 0.55
            base_basal_rate = estimated_tdd * 0.5 / 24
            pid_adjustment = -self.pid.output * 0.01
            basal_rate = float(np.clip(base_basal_rate + pid_adjustment, 0.0, 10.0))

        total_insulin_rate = basal_rate + bolus_rate

        self.previous_glucose = current_glucose
        new_glucose = self.patient.step(total_insulin_rate)  # type: ignore[union-attr]

        self.time_since_last_meal += 1
        self.time_since_last_insulin += 1
        if total_insulin_rate > 0:
            self.time_since_last_insulin = 0

        reward = self._calculate_reward(new_glucose, total_insulin_rate)
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
        return {
            "mean_glucose": float(np.mean(glucose_array)),
            "std_glucose": float(np.std(glucose_array)),
            "time_in_range_80_140": float(
                np.sum((glucose_array >= 80) & (glucose_array <= 140))
                / len(glucose_array)
                * 100
            ),
            "time_in_range_70_180": float(
                np.sum((glucose_array >= 70) & (glucose_array <= 180))
                / len(glucose_array)
                * 100
            ),
            "time_hypo_70": float(np.sum(glucose_array < 70) / len(glucose_array) * 100),
            "time_hyper_180": float(
                np.sum(glucose_array > 180) / len(glucose_array) * 100
            ),
            "total_episode_reward": self.total_episode_reward,
            "mean_insulin": float(np.mean(self.insulin_history)),
            "total_insulin": float(np.sum(self.insulin_history) / 60),
            "final_kp": self.pid.Kp,
            "final_ki": self.pid.Ki,
            "final_kd": self.pid.Kd,
            "num_boluses": len(self.bolus_history),
        }


import atexit


def _cleanup_temp_files() -> None:  # pragma: no cover - best-effort
    import shutil

    if os.path.exists("temp_data"):
        try:
            shutil.rmtree("temp_data")
        except OSError:
            pass


atexit.register(_cleanup_temp_files)
