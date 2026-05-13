"""Global RNG seeding for reproducible rollouts and training.

``set_global_seed`` covers Python's ``random``, NumPy, and TensorFlow.
TensorFlow is imported lazily so that pure-baseline workflows (which do
not need TF) can call this helper without paying TF import cost.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seed(seed: Optional[int]) -> None:
    """Set Python, NumPy, and TF RNG seeds.

    Args:
        seed: integer seed, or ``None`` to skip seeding entirely.

    Notes:
        - Sets ``PYTHONHASHSEED`` so subprocesses inherit determinism.
        - Imports TensorFlow lazily; absent TF is non-fatal.
    """
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
