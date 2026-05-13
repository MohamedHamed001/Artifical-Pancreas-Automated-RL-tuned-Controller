"""Minimal YAML config loader.

We deliberately avoid a heavy config framework. Configs are plain dicts
and callers merge them explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict (empty dict if file is empty)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a top-level mapping in {path}, got {type(data).__name__}"
        )
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge with ``override`` taking precedence."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
