from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import Dict, Any, Optional

from ap_rl.controllers.base import Controller

class GlucoseMPC(Controller):
    """
    A robust Model Predictive Controller for glucose regulation.
    Uses a 3-compartment linear state-space model to predict future glucose.

    States:
        x1: Plasma Glucose (mg/dL) - offset from target
        x2: Plasma Insulin (mU/L)
        x3: Remote Insulin Action (1/min)
    """

    def __init__(
        self,
        target_mgdl: float = 110.0,
        basal_u_h: float = 1.0,
        horizon: int = 12,  # 12 * 5 min = 1 hour
        dt: float = 5.0,     # Step size in minutes
        isf: float = 50.0,   # Insulin Sensitivity Factor (mg/dL per U)
        weight_error: float = 1.0,
        weight_input: float = 0.5,
        weight_terminal: float = 10.0
    ):
        self.target_mgdl = target_mgdl
        self.basal_u_h = basal_u_h
        self.horizon = horizon
        self.dt = dt
        self.isf = isf

        # Weights for optimization
        self.Q = weight_error
        self.R = weight_input
        self.P = weight_terminal

        # Linearized model matrices (Discrete Time)
        # These are approximations. In production, these might be
        # adaptively tuned or derived from a specific patient profile.
        # k_e: insulin elimination rate (~0.1 min^-1)
        # k_a: insulin absorption/action rate (~0.02 min^-1)
        k_e = 0.1
        k_a = 0.02

        # A matrix (3x3)
        # G[k+1]   = G[k] - k_isf * I_rem[k]
        # I_pl[k+1] = I_pl[k] - k_e * I_pl[k]
        # I_rem[k+1]= I_rem[k] - k_a * I_rem[k] + k_a * I_pl[k]

        # Sensitivity constant derived from ISF
        # Total insulin effect should match ISF over time.
        # Integral of effect over infinity should be ISF.
        # Effect = sum(dt * k_isf * I_rem)
        k_isf = (isf * k_e) / 60.0 # Approximate conversion factor

        self.A = np.eye(3)
        self.A[0, 2] = -k_isf * dt
        self.A[1, 1] = 1.0 - k_e * dt
        self.A[2, 1] = k_a * dt
        self.A[2, 2] = 1.0 - k_a * dt

        # B matrix (3x1): input is insulin infusion (U/h) -> mU/min in plasma
        # 1 U/h = 1000 mU / 60 min = 16.6 mU/min
        self.B = np.zeros((3, 1))
        self.B[1, 0] = (1000.0 / 60.0) * dt

        # Internal state estimate
        self.x_hat = np.zeros(3)

    def get_action(self, state: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Solves the MPC optimization problem.
        Assumes state[0] is current glucose in mg/dL.
        """
        g_now = state[0]

        # Current error state
        x0 = np.array([g_now - self.target_mgdl, self.x_hat[1], self.x_hat[2]])

        # Optimization variables
        u = cp.Variable((1, self.horizon))
        x = cp.Variable((3, self.horizon + 1))

        cost = 0
        constraints = [x[:, 0] == x0]

        for t in range(self.horizon):
            # Objective: 1/2 * (x' Q x + u' R u)
            cost += self.Q * cp.square(x[0, t+1])
            cost += self.R * cp.square(u[0, t] - self.basal_u_h)

            # Dynamics: x[t+1] = A x[t] + B u[t]
            constraints += [x[:, t+1] == self.A @ x[:, t] + self.B @ u[:, t]]

            # Physical constraints
            constraints += [u[0, t] >= 0.0, u[0, t] <= 15.0] # Max 15 U/h
            constraints += [x[0, t+1] >= 40.0 - self.target_mgdl] # Min 40 mg/dL

        # Terminal cost
        cost += self.P * cp.square(x[0, self.horizon])

        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # Using OSQP as it is robust and standard for MPC
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)

            if u.value is not None:
                optimal_u = float(u.value[0, 0])
                # Update internal state estimate (simple observer)
                self.x_hat = self.A @ x0 + self.B.flatten() * optimal_u
                return np.array([optimal_u], dtype=np.float32)
        except Exception:
            pass

        # Fallback to basal
        return np.array([self.basal_u_h], dtype=np.float32)

    def reset(self) -> None:
        self.x_hat = np.zeros(3)

    def update(self, reward: float, done: bool) -> None:
        pass
