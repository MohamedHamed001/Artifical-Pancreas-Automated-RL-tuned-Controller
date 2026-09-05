import numpy as np
from typing import Dict, Any, Optional

class ClinicalReward:
    """
    Implements a clinical reward function for glucose control.
    Penalizes hypo/hyperglycemia based on risk indices and safety events.
    """

    def __init__(
        self,
        hypo_weight: float = 10.0,
        hyper_weight: float = 1.0,
        insulin_weight: float = 0.1,
        safety_event_penalty: float = 5.0,
        target_range: tuple = (70, 180),
        target_glucose: float = 110.0
    ):
        self.hypo_weight = hypo_weight
        self.hyper_weight = hyper_weight
        self.insulin_weight = insulin_weight
        self.safety_event_penalty = safety_event_penalty
        self.target_range = target_range
        self.target_glucose = target_glucose

    def compute_reward(
        self,
        glucose_mgdl: float,
        insulin_delivered: float,
        safety_events: list
    ) -> float:
        """
        Calculates the reward for a single step.
        """
        # 1. Glucose Risk (Symmetric or Asymmetric)
        # We use a modified Kovatchev-style risk or a simple quadratic penalty

        error = glucose_mgdl - self.target_glucose

        if glucose_mgdl < self.target_range[0]:
            # Hypoglycemia penalty (aggressive)
            reward = -self.hypo_weight * (self.target_range[0] - glucose_mgdl)**2 / 100.0
        elif glucose_mgdl > self.target_range[1]:
            # Hyperglycemia penalty
            reward = -self.hyper_weight * (glucose_mgdl - self.target_range[1])**2 / 1000.0
        else:
            # In range: small positive reward for being close to target
            reward = 1.0 - (abs(error) / 100.0)

        # 2. Insulin penalty (to prevent over-delivery)
        reward -= self.insulin_weight * insulin_delivered

        # 3. Safety event penalty
        if safety_events:
            reward -= self.safety_event_penalty * len(safety_events)

        return float(reward)

    @staticmethod
    def blood_glucose_risk_index(glucose_mgdl: float) -> float:
        """
        Computes the Kovatchev Blood Glucose Risk Index (BGRI).
        """
        # Mapping to a symmetric space [-float, float]
        # f(G) = 1.509 * (log(G)^1.084 - 5.381)
        if glucose_mgdl <= 0:
            return 100.0 # Extreme risk

        f = 1.509 * (np.power(np.log(glucose_mgdl), 1.084) - 5.381)
        risk = 10 * f**2
        return float(risk)
