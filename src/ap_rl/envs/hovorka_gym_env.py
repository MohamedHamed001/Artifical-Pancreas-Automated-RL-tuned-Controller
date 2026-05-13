"""Experimental Gymnasium wrapper around :class:`DiabetesPIDEnv`.

Install optional dependencies::

    pip install -e ".[gym]"

This module is **not** used by the A2C trainer or the Streamlit demo; it
exists for experiments that prefer a ``gymnasium.Env`` API.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.envs.diabetes_pid_env import DiabetesPIDEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None  # type: ignore[assignment]
    spaces = None

_Base = gym.Env if gym is not None else object  # type: ignore[misc, assignment]


class HovorkaGymEnv(_Base):
    """Thin Gymnasium façade over the diabetes PID-tuning environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        patient_params: Optional[dict] = None,
        patient_weight: float = 75.0,
        target_glucose: float = 120.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if gym is None or spaces is None:  # pragma: no cover - guarded by importorskip in tests
            raise ImportError(
                'HovorkaGymEnv requires gymnasium. Install with: pip install "ap-rl[gym]"'
            )
        super().__init__()
        self.render_mode = render_mode
        params = dict(DEFAULT_PATIENT_PARAMS) if patient_params is None else patient_params
        self._env = DiabetesPIDEnv(
            patient_params=params,
            patient_weight=patient_weight,
            target_glucose=target_glucose,
            seed=seed,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.observation_space,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(self._env.action_space,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(int(seed))
        obs = self._env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action: Union[float, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict]:
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, info = self._env.step(act)
        truncated = False
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            self._env.render(mode="human")

    def close(self) -> None:
        return None
