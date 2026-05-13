"""Path helpers for ap_rl.

Resolution strategy:
1. Honour ``AP_RL_DATA_DIR`` env var if set and non-empty.
2. Walk parents looking for a ``pyproject.toml`` marker (works from src
   layout and from an editable install when scripts run inside the repo).
3. Fall back to the package's installed location (``ap_rl.__file__``) so
   that read-only installs still resolve the in-package ``configs/`` dir.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


_MARKER_FILES = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root (or best-effort fallback).

    Walks up from this file until it finds a directory containing
    ``pyproject.toml`` or ``.git``. Falls back to two parents up from
    ``src/ap_rl/utils/paths.py`` which is the in-repo layout.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        for marker in _MARKER_FILES:
            if (candidate / marker).exists():
                return candidate
    return here.parents[3]


def data_dir() -> Path:
    """Return the canonical data directory.

    Order of precedence:
    1. ``AP_RL_DATA_DIR`` environment override.
    2. ``<repo_root>/data/test_scenarios/`` (preferred new layout).
    3. ``<repo_root>/TestData/`` (legacy layout retained as fallback).
    """
    env_override = os.environ.get("AP_RL_DATA_DIR", "").strip()
    if env_override:
        path = Path(env_override).expanduser().resolve()
        if path.exists():
            return path

    root = repo_root()
    new_layout = root / "data" / "test_scenarios"
    if new_layout.exists():
        return new_layout

    legacy = root / "TestData"
    return legacy


def checkpoints_dir() -> Path:
    """Return the checkpoints directory, creating it if missing."""
    path = repo_root() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configs_dir() -> Path:
    """Return the ``configs/`` directory at the repository root."""
    return repo_root() / "configs"
