"""Synthetic patient-profile loader.

Profiles live in ``configs/profiles/*.yaml`` and combine three concerns:

1. Hovorka parameter overrides (``patient_params``) - applied on top of
   :data:`ap_rl.training.train_a2c.DEFAULT_PATIENT_PARAMS`.
2. Clinical insulin-calculator overrides (``carb_ratio``, ``isf``).
3. Demo metadata (display name, description, observation noise).

These profiles are explicitly **synthetic** and only intended for
research / demo use, not clinical decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ap_rl.utils.config import deep_merge, load_yaml
from ap_rl.utils.paths import configs_dir


@dataclass
class PatientProfile:
    """In-memory representation of a synthetic patient profile."""

    name: str
    display_name: str
    description: str = ""
    patient_params: dict = field(default_factory=dict)
    patient_weight: float = 75.0
    target_glucose: float = 120.0
    carb_ratio: Optional[float] = None
    isf: Optional[float] = None
    observation_noise_std: float = 0.0


def load_profile(
    name: str,
    base_patient_params: Optional[dict] = None,
    profiles_dir: Optional[Path] = None,
) -> PatientProfile:
    """Load a profile YAML by name (without extension).

    Args:
        name: Profile filename stem (e.g. ``"insulin_sensitive"``).
        base_patient_params: Optional default Hovorka dict to merge
            overrides onto. When ``None``, callers should supply their
            own defaults; we do **not** import training defaults here to
            avoid circular dependencies.
        profiles_dir: Optional path override. Defaults to
            ``<configs>/profiles``.
    """
    profiles_path = (
        Path(profiles_dir) if profiles_dir is not None else configs_dir() / "profiles"
    )
    yaml_path = profiles_path / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Profile YAML not found: {yaml_path}")

    cfg = load_yaml(yaml_path)
    patient_params = dict(base_patient_params or {})
    patient_params = deep_merge(patient_params, cfg.get("patient_params", {}))

    insulin_cfg = cfg.get("insulin", {}) or {}

    return PatientProfile(
        name=name,
        display_name=cfg.get("display_name", name.replace("_", " ").title()),
        description=cfg.get("description", ""),
        patient_params=patient_params,
        patient_weight=float(cfg.get("patient_weight", 75.0)),
        target_glucose=float(cfg.get("target_glucose", 120.0)),
        carb_ratio=insulin_cfg.get("carb_ratio"),
        isf=insulin_cfg.get("isf"),
        observation_noise_std=float(cfg.get("observation_noise_std", 0.0)),
    )


def list_profiles(profiles_dir: Optional[Path] = None) -> list[str]:
    """Return the available profile names (yaml stems)."""
    profiles_path = (
        Path(profiles_dir) if profiles_dir is not None else configs_dir() / "profiles"
    )
    if not profiles_path.exists():
        return []
    return sorted(p.stem for p in profiles_path.glob("*.yaml"))
