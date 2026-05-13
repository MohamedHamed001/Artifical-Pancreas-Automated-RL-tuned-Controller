"""Simulation environments for ap_rl.

* :class:`HovorkaPatient` - the underlying Type-1 diabetes ODE model.
* :class:`DiabetesPIDEnv` - 13-D observation / 3-D PID-delta action env
  used by the A2C agent.
* :class:`HovorkaGymEnv` - **optional** Gymnasium wrapper (``pip install -e ".[gym]"``).
* :func:`scenario_builder.build_scenario` - generate deterministic meal +
  exercise schedules from YAML templates.
"""

from __future__ import annotations

from ap_rl.envs.diabetes_pid_env import DiabetesPIDEnv
from ap_rl.envs.hovorka_gym_env import HovorkaGymEnv
from ap_rl.envs.hovorka_patient import HovorkaPatient

__all__ = ["HovorkaPatient", "DiabetesPIDEnv", "HovorkaGymEnv"]
