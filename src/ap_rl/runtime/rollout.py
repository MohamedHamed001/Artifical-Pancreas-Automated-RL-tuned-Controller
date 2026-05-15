"""Single-episode rollout helper.

This module is intentionally framework-agnostic: no Streamlit, no TF
imports unless ``controller='rl'`` is requested. The demo and the smoke
scripts both call :func:`run_episode`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ap_rl.agents.diabetes_a2c_actor import DiabetesActor
from ap_rl.envs import DiabetesPIDEnv
from ap_rl.utils.checkpoint_filenames import ACTOR_BEST, actor_load_candidates
from ap_rl.utils.paths import checkpoints_dir


@dataclass
class EpisodeRecord:
    """Compact record of a finished episode for plotting / metrics."""

    times: list[float] = field(default_factory=list)
    glucose: list[float] = field(default_factory=list)
    insulin: list[float] = field(default_factory=list)
    basal: list[float] = field(default_factory=list)
    bolus: list[float] = field(default_factory=list)
    Kp: list[float] = field(default_factory=list)
    Ki: list[float] = field(default_factory=list)
    Kd: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    meals: list[dict] = field(default_factory=list)
    exercise: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    controller: str = "baseline"
    target_glucose: float = 120.0

    def as_arrays(self):
        """Convert lists to numpy arrays for vectorised plotting."""
        return {
            "times": np.asarray(self.times),
            "glucose": np.asarray(self.glucose),
            "insulin": np.asarray(self.insulin),
            "basal": np.asarray(self.basal),
            "bolus": np.asarray(self.bolus),
            "Kp": np.asarray(self.Kp),
            "Ki": np.asarray(self.Ki),
            "Kd": np.asarray(self.Kd),
            "rewards": np.asarray(self.rewards),
        }


def _probe_state_dim(actor_path: str) -> int:
    """Read the first Dense kernel shape from an HDF5/h5 checkpoint.

    Returns the input dimension (rows of the first kernel), or 19 as the
    safe default if the file cannot be inspected.  This lets the rollout
    always build an actor whose architecture matches the saved weights,
    even when the observation space has been extended between training runs.
    """
    try:
        import h5py  # ships with tensorflow; always available if TF is installed
        with h5py.File(actor_path, "r") as f:
            # Keras saves weights under 'layers/dense/vars/0' (Keras 3) or
            # '_layer_checkpoint_dependencies/dense/vars/0' (legacy).
            # Walk all datasets looking for the first 2-D one.
            def _first_2d_shape(group):
                for key in group:
                    item = group[key]
                    if hasattr(item, "shape") and len(item.shape) == 2:
                        return item.shape  # (in_dim, out_dim)
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
    state_dim: Optional[int] = None,   # None = auto-detect from checkpoint
    action_dim: int = 3,
    action_bound: float = 0.1,
    actor_filename: str = ACTOR_BEST,
    checkpoints_path: Optional[str | os.PathLike] = None,
):
    """Try to load the actor network from ``checkpoints/``.

    The ``state_dim`` is **auto-detected from the checkpoint file** when not
    supplied, so the actor architecture always matches the saved weights — even
    after the observation space was extended (e.g. 16-D → 19-D).

    Returns the actor or ``None`` if the file is missing. The caller is
    responsible for handling the fallback (e.g. running the baseline
    controller instead).
    """
    ckpt_dir = (
        os.fspath(checkpoints_path) if checkpoints_path is not None else os.fspath(checkpoints_dir())
    )
    for actor_path in actor_load_candidates(actor_filename, ckpt_dir):
        if not os.path.exists(actor_path):
            continue

        # Auto-detect the correct input dimension from the checkpoint file.
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


def run_episode(
    env: DiabetesPIDEnv,
    controller: str = "baseline",
    max_steps: Optional[int] = None,
    actor=None,
    seed: Optional[int] = None,
) -> EpisodeRecord:
    """Run a single episode and return an :class:`EpisodeRecord`.

    Args:
        env: A :class:`DiabetesPIDEnv` instance. Reset internally.
        controller: ``"baseline"`` (zero-delta PID) or ``"rl"`` (use the
            supplied ``actor``). Baseline never imports TF.
        max_steps: optional override for ``env.max_episode_length``.
        actor: optional preloaded actor for ``controller="rl"``. If
            ``None`` and RL requested, falls back to baseline.
        seed: optional seed forwarded to the env before reset.

    Returns:
        Populated :class:`EpisodeRecord`.
    """
    if seed is not None:
        env.seed(seed)

    state = env.reset()
    record = EpisodeRecord(controller=controller, target_glucose=env.target_glucose)
    record.meals = list(env.meal_data)
    record.exercise = list(env.exercise_data)

    horizon = max_steps if max_steps is not None else env.max_episode_length

    use_rl = controller == "rl" and actor is not None

    step = 0
    done = False
    while not done and step < horizon:
        if use_rl:
            # Handle state dimension mismatch (e.g. env returns 19-D but actor is 16-D)
            # by truncating the state vector. This allows old models to run in the
            # upgraded environment.
            truncated_state = state[:actor.state_dim]
            action = actor.get_action(truncated_state)
        else:
            action = np.zeros(env.action_space, dtype=np.float32)

        state, reward, done, info = env.step(action)

        record.times.append(env.patient.time)  # type: ignore[union-attr]
        record.glucose.append(info["glucose"])
        record.insulin.append(info["total_insulin"])
        record.basal.append(info["basal_insulin"])
        record.bolus.append(info["bolus_insulin"])
        record.Kp.append(info["Kp"])
        record.Ki.append(info["Ki"])
        record.Kd.append(info["Kd"])
        record.rewards.append(reward)
        step += 1

    record.stats = env.get_statistics()
    return record
