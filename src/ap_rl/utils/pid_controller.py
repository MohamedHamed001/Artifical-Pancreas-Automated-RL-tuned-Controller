"""Simple PID controller (IvPID, GPL-3).

Original author: Caner Durmusoglu (Ivmech Mechatronics Ltd., 2015).
Vendored here so the package has no external runtime dependency on the
upstream ``ivPID`` library and so we can keep the diabetes-specific
behaviour (no sample-time gating during ``update``).
"""

from __future__ import annotations

import time
from typing import Optional


class PID:
    """Discrete-time PID controller used by :class:`DiabetesPIDEnv`.

    The controller is intentionally minimal:
      ``output = Kp * P + Ki * I + Kd * D``

    Behaviour preserved verbatim from the legacy implementation so that
    pre/post-refactor numerical comparisons are identical.
    """

    def __init__(
        self,
        P: float = 1.2,
        I: float = 1.0,
        D: float = 0.0001,
        current_time: float = 0.0,
    ) -> None:
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.0
        self.current_time = current_time
        self.last_time = self.current_time

        self.clear()

    def clear(self) -> None:
        """Reset accumulator state and gains-independent terms."""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value: float, current_time: Optional[float] = None) -> None:
        """Compute the PID output for the latest feedback sample."""
        error = self.SetPoint - feedback_value

        if current_time is not None:
            self.current_time = current_time
        else:
            self.current_time += 1.0  # fallback to 1-min increments

        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        self.PTerm = self.Kp * error
        self.ITerm += error * delta_time

        if self.ITerm < -self.windup_guard:
            self.ITerm = -self.windup_guard
        elif self.ITerm > self.windup_guard:
            self.ITerm = self.windup_guard

        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        self.last_time = self.current_time
        self.last_error = error

        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain: float) -> None:
        self.Kp = proportional_gain

    def setKi(self, integral_gain: float) -> None:
        self.Ki = integral_gain

    def setKd(self, derivative_gain: float) -> None:
        self.Kd = derivative_gain

    def setWindup(self, windup: float) -> None:
        """Set the symmetric integrator clamp."""
        self.windup_guard = windup

    def setSampleTime(self, sample_time: float) -> None:
        self.sample_time = sample_time
