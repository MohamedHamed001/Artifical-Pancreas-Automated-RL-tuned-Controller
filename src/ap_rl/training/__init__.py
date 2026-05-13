"""Training entry points for ap_rl.

The heavy lifting lives in :mod:`ap_rl.training.train_a2c`; that module
lazy-imports TensorFlow so the demo / smoke scripts can import
``ap_rl.training.train_a2c.DEFAULT_PATIENT_PARAMS`` without TF.
"""

from __future__ import annotations

__all__ = ["main", "train_a2c"]


def __getattr__(name):  # pragma: no cover - simple lazy passthrough
    if name in {"main", "train_a2c"}:
        from ap_rl.training import train_a2c as _module

        return _module.main if name == "main" else _module.train
    raise AttributeError(name)
