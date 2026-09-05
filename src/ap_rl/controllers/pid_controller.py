from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from ap_rl.utils.pid_controller import PID
from ap_rl.controllers.base import Controller

class PIDController(Controller):
    """
    Standard PID Controller for direct glucose regulation.
    Input: state (glucose), Output: insulin dose (U/h).
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        target_mgdl: float = 120.0,
        basal_u_h: float = 1.0
    ):
        self.pid = PID(P=Kp, I=Ki, D=Kd)
        self.pid.SetPoint = target_mgdl
        self.basal_u_h = basal_u_h
        self.target_mgdl = target_mgdl
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    @property
    def Kp(self): return self.pid.Kp
    @Kp.setter
    def Kp(self, val): self.pid.Kp = val

    @property
    def Ki(self): return self.pid.Ki
    @Ki.setter
    def Ki(self, val): self.pid.Ki = val

    @property
    def Kd(self): return self.pid.Kd
    @Kd.setter
    def Kd(self, val): self.pid.Kd = val

    def get_action(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Compute direct insulin dose.
        Assumes state[0] is glucose in mg/dL.
        """
        glucose = state[0]
        self.pid.update(glucose)

        # PID output is typically used as a delta or adjustment to basal
        pid_adjustment = -self.pid.output * 0.01  # Legacy scaling factor
        dose = float(np.clip(self.basal_u_h + pid_adjustment, 0.0, 10.0))

        return np.array([dose], dtype=np.float32)

    def reset(self) -> None:
        self.pid.clear()
        self.pid.SetPoint = self.target_mgdl

    def update(self, reward: float, done: bool) -> None:
        pass


class SupervisoryController(Controller):
    """
    A controller that uses an RL agent to tune a low-level PID controller.
    """

    def __init__(
        self,
        agent: Any,
        inner_pid: PIDController,
        state_dim: int = 19
    ):
        self.agent = agent
        self.pid = inner_pid
        self.state_dim = state_dim

    def get_action(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        # 1. Get gain adjustments from RL agent
        # State might be full 19-D or just the first few
        agent_state = state
        if agent_state.shape[0] > self.state_dim:
            agent_state = agent_state[:self.state_dim]

        # Call agent (e.g. A2C Actor)
        # Note: We expect (dKp, dKi, dKd)
        if hasattr(self.agent, "get_action"):
            deltas = self.agent.get_action(agent_state)
        elif hasattr(self.agent, "predict"):
            deltas, _ = self.agent.predict(agent_state)
        else:
            deltas = np.zeros(3, dtype=np.float32)

        # 2. Apply adjustments to PID
        dkp, dki, dkd = deltas
        self.pid.Kp = float(np.clip(self.pid.Kp + dkp * 0.2, 0.01, 10.0))
        self.pid.Ki = float(np.clip(self.pid.Ki + dki * 0.1, 0.0, 1.0))
        self.pid.Kd = float(np.clip(self.pid.Kd + dkd * 0.1, 0.0, 1.0))

        # 3. Compute insulin using updated PID
        # state[0] is glucose
        return self.pid.get_action(state, info)

    def reset(self) -> None:
        self.pid.reset()

    def update(self, reward: float, done: bool) -> None:
        if hasattr(self.agent, "update"):
            self.agent.update(reward, done)
