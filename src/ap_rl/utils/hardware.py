"""Hardware configuration helpers for maximum training throughput.

Call :func:`configure_for_training` once at the top of any training
script **before** building any TensorFlow models.  It is safe to call
multiple times (idempotent).

M4 Mac specifics
----------------
* TF 2.21 (standard PyPI) does not expose the Metal GPU on Apple Silicon;
  Metal support requires the ``tensorflow-macos`` + ``tensorflow-metal``
  stack which is capped at TF 2.16.  We therefore focus on maximising
  CPU throughput, which on the M4's 10-core layout is substantial.
* ``inter_op_parallelism_threads`` controls how many independent TF ops
  run in parallel (set to physical cores).
* ``intra_op_parallelism_threads`` controls how many threads are used
  inside a single op (matrix mul etc.).  Setting this to logical-core
  count saturates the CPU pipeline.
* ``tf.function`` JIT-compilation (XLA tracing) amortises Python overhead
  across repeated calls — the actor/critic update step is called ~45 k
  times per training run, so even a 2× kernel speedup compounds heavily.
"""

from __future__ import annotations

import os
import multiprocessing


def _cpu_counts() -> tuple[int, int]:
    """Return (physical_cores, logical_cores)."""
    logical = multiprocessing.cpu_count()
    # On Apple Silicon sysctl is the reliable source; fall back to logical count.
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True, text=True, timeout=2
        )
        physical = int(result.stdout.strip())
    except Exception:
        physical = max(1, logical // 2)
    return physical, logical


def configure_for_training(verbose: bool = True) -> dict:
    """Configure TensorFlow for maximum CPU throughput on the current machine.

    Must be called **before** any ``tf.keras`` model is built.

    Returns a dict with the resolved configuration for logging.
    """
    import tensorflow as tf

    physical, logical = _cpu_counts()

    # Thread counts: physical for inter-op, logical for intra-op.
    # This maximises both operation-level and kernel-level parallelism.
    tf.config.threading.set_inter_op_parallelism_threads(physical)
    tf.config.threading.set_intra_op_parallelism_threads(logical)

    # Tell NumPy-based code to use all cores too
    os.environ.setdefault("OMP_NUM_THREADS", str(logical))
    os.environ.setdefault("MKL_NUM_THREADS", str(logical))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(logical))

    # Suppress noisy TF logs (keep errors only)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # GPU memory growth (no-op when no GPU, harmless)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised

    cfg = {
        "physical_cores": physical,
        "logical_cores": logical,
        "inter_op_threads": physical,
        "intra_op_threads": logical,
        "gpu_devices": [g.name for g in gpus],
    }

    if verbose:
        print(
            f"[hardware] M4 CPU: {physical} physical / {logical} logical cores — "
            f"inter_op={physical}, intra_op={logical}"
        )
        if gpus:
            print(f"[hardware] GPU(s): {cfg['gpu_devices']}")
        else:
            print("[hardware] No GPU detected — running on CPU (Metal requires tensorflow-macos≤2.16)")

    return cfg
