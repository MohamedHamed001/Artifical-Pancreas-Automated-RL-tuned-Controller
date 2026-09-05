from typing import Protocol, Dict, Any, Optional
import numpy as np


class Controller(Protocol):
    """
    Abstract Protocol for all pancreas controllers (PID, RL, MPC).
    Ensures they all expose a unified interface for the runtime loop.
    """

    def get_action(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Compute the next control action (delta_gains or direct insulin).

        Args:
            state: The observation vector from the environment.
            info: Optional metadata from the environment.

        Returns:
            The action vector (np.ndarray).
        """
        ...

    def reset(self) -> None:
        """Reset internal controller state."""
        ...

    def update(self, reward: float, done: bool) -> None:
        """
        Optional learning update for RL controllers.
        For rule-based controllers, this is a no-op.
        """
        ...
