from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import numpy as np


@dataclass
class StepRecord:
    """
    Data captured at each discrete simulation step (e.g., every 1 or 5 minutes).
    """
    time: float  # Simulation time in minutes
    true_glucose: float
    observed_glucose: float
    requested_insulin: float
    delivered_insulin: float
    basal: float
    bolus: float
    iob: float
    cob: float
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    safety_events: List[str] = field(default_factory=list)
    controller_metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_events: List[str] = field(default_factory=list)
    reward: float = 0.0


@dataclass
class EpisodeRecord:
    """
    Summary of a full simulation run.
    """
    steps: List[StepRecord] = field(default_factory=list)
    patient_id: str = "unknown"
    scenario_id: str = "unknown"
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Legacy fields for plotting compatibility
    controller_name: str = "baseline"
    target_glucose: float = 120.0
    meals: List[Dict[str, Any]] = field(default_factory=list)
    exercise: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def controller(self) -> str:
        """Alias for controller_name."""
        return self.controller_name

    @property
    def stats(self) -> Dict[str, Any]:
        """Alias for metadata["stats"]."""
        return self.metadata.get("stats", {})

    @property
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.steps])

    @property
    def glucose(self) -> np.ndarray:
        return np.array([s.true_glucose for s in self.steps])

    @property
    def insulin(self) -> np.ndarray:
        return np.array([s.delivered_insulin for s in self.steps])

    @property
    def basal(self) -> np.ndarray:
        return np.array([s.basal for s in self.steps])

    @property
    def bolus(self) -> np.ndarray:
        return np.array([s.bolus for s in self.steps])

    @property
    def Kp(self) -> np.ndarray:
        return np.array([s.Kp for s in self.steps])

    @property
    def Ki(self) -> np.ndarray:
        return np.array([s.Ki for s in self.steps])

    @property
    def Kd(self) -> np.ndarray:
        return np.array([s.Kd for s in self.steps])

    @property
    def rewards(self) -> np.ndarray:
        return np.array([s.reward for s in self.steps])

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Compatibility method for legacy plotting."""
        return {
            "times": self.times,
            "glucose": self.glucose,
            "insulin": self.insulin,
            "basal": self.basal,
            "bolus": self.bolus,
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "rewards": self.rewards,
        }
