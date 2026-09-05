import numpy as np
from typing import Callable

def rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float
) -> np.ndarray:
    """
    Classic fourth-order Runge-Kutta step.

    Args:
        f: Right-hand side of the ODE, dy/dt = f(t, y)
        t: Current simulation time
        y: Current state vector
        h: Step size

    Returns:
        Next state vector y(t + h)
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)

    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
