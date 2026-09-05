"""Single-episode rollout helper.

This module is intentionally framework-agnostic: no Streamlit, no TF
imports at top level. The demo and the smoke scripts both call :func:`run_episode`.
"""

from __future__ import annotations

import os
from typing import Optional, Union, Any

import numpy as np

from ap_rl.envs import DiabetesPIDEnv
from ap_rl.controllers.base import Controller
from ap_rl.controllers.pid_controller import PIDController
from ap_rl.utils.checkpoint_filenames import ACTOR_BEST, actor_load_candidates
from ap_rl.utils.paths import checkpoints_dir
from ap_rl.core.records import StepRecord, EpisodeRecord
from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.core.types import PatientConfig


def _probe_state_dim(actor_path: str) -> int:
    """Read the first Dense kernel shape from an HDF5/h5 checkpoint."""
    try:
        import h5py  # ships with tensorflow; always available if TF is installed
        with h5py.File(actor_path, "r") as f:
            def _first_2d_shape(group):
                for key in group:
                    item = group[key]
                    if hasattr(item, "shape") and len(item.shape) == 2:
                        return item.shape
                    if hasattr(item, "keys"):
                        result = _first_2d_shape(item)
                        if result is not None:
                            return result
                return None

            shape = _first_2d_shape(f)
            if shape is not None:
                return int(shape[0])
    except Exception:
        pass
    return 19  # safe default for the new 19-D architecture


def load_actor_from_checkpoints(
    state_dim: Optional[int] = None,
    action_dim: int = 3,
    action_bound: float = 0.1,
    actor_filename: str = ACTOR_BEST,
    checkpoints_path: Optional[str | os.PathLike] = None,
):
    """Try to load the actor network from ``checkpoints/``."""
    # Lazy import to avoid TF dependency for baseline runs
    from ap_rl.agents.diabetes_a2c_actor import DiabetesActor

    ckpt_dir = (
        os.fspath(checkpoints_path) if checkpoints_path is not None else os.fspath(checkpoints_dir())
    )
    for actor_path in actor_load_candidates(actor_filename, ckpt_dir):
        if not os.path.exists(actor_path):
            continue

        detected_dim = _probe_state_dim(actor_path)
        resolved_dim = state_dim if state_dim is not None else detected_dim

        actor = DiabetesActor(
            state_dim=resolved_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            learning_rate=1e-4,
        )
        actor.load_weights(actor_path)
        return actor
    return None


class ZeroTuner(Controller):
    """A controller that returns zero adjustments (for baseline tuning runs)."""
    def get_action(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    def reset(self): pass
    def update(self, reward: float, done: bool) -> None: pass

def run_episode(
    env: DiabetesPIDEnv,
    controller: Union[Controller, str] = "baseline",
    max_steps: Optional[int] = None,
    actor: Optional[Any] = None,
    seed: Optional[int] = None,
) -> EpisodeRecord:
    """
    Run a single simulation episode.
    """
    if seed is not None:
        env.seed(seed)

    state_tuple = env.reset()
    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple

    # Initialize record
    record = EpisodeRecord(
        controller_name=str(controller),
        target_glucose=env.target_glucose,
        meals=list(env.meal_data) if hasattr(env, "meal_data") else [],
        exercise=list(env.exercise_data) if hasattr(env, "exercise_data") else []
    )

    horizon = max_steps if max_steps is not None else env.max_episode_length

    # Legacy controller wrapping
    if isinstance(controller, str):
        if controller == "rl" and actor is not None:
            active_controller = actor
        elif controller == "baseline" and isinstance(env, DiabetesPIDEnv):
            active_controller = ZeroTuner()
        else:
            active_controller = PIDController(
                target_mgdl=env.target_glucose,
                basal_u_h=env.patient.basal_rate_uh,
                Kp=env.pid.Kp,
                Ki=env.pid.Ki,
                Kd=env.pid.Kd
            )
    else:
        active_controller = controller

    active_controller.reset()
    step_idx = 0
    done = False
    info = {}

    while not done and step_idx < horizon:
        state_to_use = state
        if hasattr(active_controller, "state_dim"):
            state_to_use = state[:active_controller.state_dim]

        action = active_controller.get_action(state_to_use, info)

        step_res = env.step(action)
        if len(step_res) == 5:
            state, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            state, reward, done, info = step_res
        active_controller.update(reward, done)

        # Create StepRecord
        s_record = StepRecord(
            time=float(env.patient.time),
            true_glucose=float(info["glucose"]),
            observed_glucose=float(info["glucose"]),
            requested_insulin=float(info.get("requested_insulin", info["total_insulin"])),
            delivered_insulin=float(info["total_insulin"]),
            basal=float(info["basal_insulin"]),
            bolus=float(info["bolus_insulin"]),
            iob=float(info.get("iob", 0.0)),
            cob=float(info.get("cob", 0.0)),
            Kp=float(info["Kp"]),
            Ki=float(info["Ki"]),
            Kd=float(info["Kd"]),
            reward=float(reward),
            safety_events=info.get("safety_events", [])
        )
        record.steps.append(s_record)
        step_idx += 1

    record.total_reward = sum(s.reward for s in record.steps)
    record.metadata["stats"] = env.get_statistics()
    return record


def run_simulation(
    patient_config: PatientConfig,
    controller: Controller,
    meals: List[Dict[str, Any]],
    exercise: List[Dict[str, Any]],
    duration_min: int = 1440,
    dt_min: int = 5
) -> EpisodeRecord:
    """
    Run a simulation using the modular SimulationRunner.
    """
    config = SimulationConfig(
        patient_config=patient_config,
        controller=controller,
        duration_min=duration_min,
        dt_min=dt_min,
        target_glucose_mgdl=patient_config.params.get("G_target", 120.0),
        basal_rate_u_h=patient_config.params.get("U_basal", 1.0)
    )
    runner = SimulationRunner(config)
    return runner.run(meal_data=meals, exercise_data=exercise)
