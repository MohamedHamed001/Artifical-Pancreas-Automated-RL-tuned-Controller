"""A2C agent package.

* :class:`DiabetesActor` - actor network producing ``(dKp, dKi, dKd)``.
* :class:`DiabetesCritic` - critic state-value head.
* :class:`DiabetesA2CAgent` - orchestrator with train/test/save APIs.

These names are exposed via :pep:`562` lazy ``__getattr__`` so the
package can be imported even when TensorFlow is not installed (baseline
PID rollouts and meal/insulin tests do not need TF).
"""

from __future__ import annotations

__all__ = ["DiabetesActor", "DiabetesCritic", "DiabetesA2CAgent"]


def __getattr__(name):  # pragma: no cover - simple lazy passthrough
    if name == "DiabetesActor":
        from ap_rl.agents.diabetes_a2c_actor import DiabetesActor

        return DiabetesActor
    if name == "DiabetesCritic":
        from ap_rl.agents.diabetes_a2c_critic import DiabetesCritic

        return DiabetesCritic
    if name == "DiabetesA2CAgent":
        from ap_rl.agents.diabetes_a2c_agent import DiabetesA2CAgent

        return DiabetesA2CAgent
    raise AttributeError(name)
