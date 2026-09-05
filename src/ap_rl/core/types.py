from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol, Union, TYPE_CHECKING


@dataclass(frozen=True)
class PatientState:
    """Full physiological state of the patient model."""
    time_min: int
    glucose_mgdl: float
    glucose_rate_mgdl_min: float
    iob_u: float
    cob_g: float
    exercise_active: bool
    compartments: Dict[str, float]


@dataclass(frozen=True)
class SensorReading:
    """Observation from a CGM or ideal sensor."""
    time_min: int
    glucose_mgdl: float
    is_missing: bool = False
    noise_std_mgdl: Optional[float] = None


@dataclass(frozen=True)
class InsulinCommand:
    """Requested or delivered insulin action."""
    time_min: int
    basal_u_h: float
    bolus_u: float = 0.0
    total_u_h: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyDecision:
    """Outcome of safety supervisor evaluation."""
    requested: InsulinCommand
    delivered: InsulinCommand
    is_modified: bool
    active_constraints: List[str]
    predicted_min_glucose_mgdl: Optional[float] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    """Complete trace of one simulation step."""
    time_min: int
    true_state: PatientState
    sensor: SensorReading
    requested_command: InsulinCommand
    delivered_command: InsulinCommand
    safety: SafetyDecision
    reward: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PatientConfig:
    """Static parameters defining a virtual patient."""
    name: str
    params: Dict[str, Any]
    body_weight_kg: float = 75.0


@dataclass(frozen=True)
class Scenario:
    """Sequence of exogenous events (meals, exercise) for a simulation."""
    id: str
    meals: List[Dict[str, Any]] = field(default_factory=list)
    exercise: List[Dict[str, Any]] = field(default_factory=list)
    duration_min: int = 1440

    # High-performance lookup tables [time, value]
    meal_data: Optional[np.ndarray] = None
    exercise_data: Optional[np.ndarray] = None
