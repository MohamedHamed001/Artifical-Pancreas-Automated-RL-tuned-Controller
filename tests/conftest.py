"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Headless matplotlib for the publication smoke test.
os.environ.setdefault("MPLBACKEND", "Agg")
