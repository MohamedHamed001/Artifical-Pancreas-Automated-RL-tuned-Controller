import numpy as np
from typing import Any, Callable, Dict
from ap_rl.simulation.integrators import rk4_step

DEFAULT_HOVORKA_PARAMS: Dict[str, float] = {
    "k_a1": 0.006,
    "k_a2": 0.06,
    "k_a3": 0.03,
    "k_b1": 0.03072,
    "k_b2": 0.0492,
    "k_b3": 1.56,
    "V_I": 0.12,
    "t_max_I": 55.0,
    "k_e": 0.138,
    "F_01": 0.0097,
    "V_G": 0.16,
    "k_12": 0.066,
    "EGP_0": 0.0161,
    "AG": 0.8,
    "t_max_G": 40.0,
    "A_EGP": 0.0,
    "phi_EGP": -60.0,
    "F_peak": 1.0,
    "K_rise": 5.0 / 60.0,
    "K_decay": 0.01,
    "G_thresh": 9.0,
    "k_R": 0.0031,
}

# Try to import numba for acceleration
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(f): return f
        return decorator


@njit(cache=True)
def circadian_multiplier(
    time_min: float, amplitude: float, phase_min: float
) -> float:
    return 1.0 + amplitude * np.sin(
        2.0 * np.pi * (time_min - phase_min) / 1440.0
    )


@njit(fastmath=True, cache=True)
def hovorka_ode(
    t: float,
    y: np.ndarray,
    u_i_min: float,
    u_g_g_min: float,
    f_sens: float,
    p: np.ndarray
) -> np.ndarray:
    """
    RHS for the 10-state Hovorka model.

    Args:
        t: Time in minutes
        y: State vector [S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2]
        u_i_min: Insulin infusion rate (U/min)
        u_g_g_min: External glucose input to the gut compartment (g/min)
        f_sens: Exercise sensitivity multiplier
        p: Parameter array
    """
    S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2 = y

    (
        k_a1, k_a2, k_a3, k_b1, k_b2, k_b3,
        V_I, t_max_I, k_e, F_01, V_G, k_12,
        EGP_0, AG, t_max_G, A_EGP, phi_EGP,
        G_thresh, k_R
    ) = p

    # Insulin sub-model
    dS1 = u_i_min - (S1 / t_max_I)
    dS2 = (S1 - S2) / t_max_I
    dI = (S2 / (t_max_I * V_I)) - k_e * I

    # Insulin action sub-model
    dx1 = f_sens * k_b1 * I - k_a1 * x1
    dx2 = f_sens * k_b2 * I - k_a2 * x2
    dx3 = f_sens * k_b3 * I - k_a3 * x3

    # Glucose sub-model
    glucose = Q1 / V_G if V_G > 0 else 0.0

    # Circadian EGP
    egp_t = EGP_0 * circadian_multiplier(t, A_EGP, phi_EGP)
    egp = egp_t * (1.0 - x3)

    # Renal clearance
    if glucose > G_thresh:
        f_r = k_R * (glucose - G_thresh) * V_G
    else:
        f_r = 0.0

    # Non-insulin-dependent glucose consumption
    if glucose < 4.5:
        f_01_total = F_01 * glucose / 4.5
    else:
        f_01_total = F_01

    # U_id is glucose appearance from gut (D2 is now in mmol)
    u_id = (AG * D2) / t_max_G

    dQ1 = u_id + egp - f_r - f_01_total - x1 * Q1 + k_12 * Q2
    dQ2 = x1 * Q1 - (k_12 + x2) * Q2

    # Meal sub-model (convert input g to mmol)
    dD1 = (u_g_g_min / 0.180182) - D1 / t_max_G
    dD2 = (D1 - D2) / t_max_G

    return np.array([dS1, dS2, dI, dx1, dx2, dx3, dQ1, dQ2, dD1, dD2])


def pack_params(p_dict: Dict[str, Any], body_weight: float) -> np.ndarray:
    """Pack dictionary parameters into a Numba-friendly array."""
    params = {**DEFAULT_HOVORKA_PARAMS, **p_dict}
    return np.array([
        params["k_a1"],
        params["k_a2"],
        params["k_a3"],
        params["k_b1"],
        params["k_b2"],
        params["k_b3"],
        params["V_I"] * body_weight,
        params["t_max_I"],
        params["k_e"],
        params["F_01"] * body_weight,
        params["V_G"] * body_weight,
        params["k_12"],
        params["EGP_0"] * body_weight,
        params["AG"],
        params["t_max_G"],
        params["A_EGP"],
        params["phi_EGP"],
        params["G_thresh"],
        params["k_R"],
    ], dtype=np.float64)


def hovorka_step(
    state: np.ndarray,
    t: float,
    u_i_min: float,
    f_sens_at: Callable[[float], float],
    p_array: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Advance Hovorka state by dt minutes using RK4 and non-negativity guards.
    """
    def rhs(curr_t, curr_y):
        return hovorka_ode(
            curr_t, curr_y, u_i_min, 0.0, f_sens_at(curr_t), p_array
        )

    new_state = rk4_step(rhs, t, state, dt)

    # Non-negativity guards
    new_state = np.maximum(new_state, 0.0)

    return new_state
