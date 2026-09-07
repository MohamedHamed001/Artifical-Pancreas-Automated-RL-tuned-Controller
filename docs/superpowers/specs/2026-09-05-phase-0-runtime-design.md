# Phase 0C Reproducible Runtime Design

## Goal

Make development, testing, and the temporary compatibility container resolve
the same Python dependency graph before scientific behavior is changed.

## Decisions

- Python 3.11 is the only validated minor release during the stabilization
  phases. Package metadata must require `>=3.11,<3.12`, and `.python-version`
  must select `3.11`.
- `pyproject.toml` is the only hand-maintained dependency source of truth.
- `uv.lock` records the exact resolved dependency graph and is committed.
- The redundant hand-maintained `requirements.txt` is removed. It may return
  later only as generated output for a platform that requires it.
- Gymnasium remains a base dependency because `ap_rl.envs` imports it
  unconditionally.
- TensorFlow and `h5py` move to the `rl` extra.
- CVXPY moves to the `mpc` extra.
- Streamlit and Plotly remain in the `demo` extra only while the deprecated UI
  is retained.
- Numba remains in the `fast` extra.
- pytest and pytest-mock remain in the `dev` extra.
- A full contributor/test environment uses all extras. Minimal simulation
  users install only the base project.
- The Docker image continues to run the deprecated Streamlit app temporarily,
  but installs reproducibly from `uv.lock` with only `demo` and `mpc` extras.
- A `.dockerignore` excludes Git data, virtual environments, caches,
  checkpoints, scratch files, and generated simulation artifacts from the
  Docker build context.

## Dependency boundaries

```text
base: numpy, scipy, pandas, matplotlib, pyyaml, requests, gymnasium
rl: tensorflow, h5py
mpc: cvxpy
demo: streamlit, plotly
fast: numba
dev: pytest, pytest-mock
```

## Verification

The lockfile must resolve for Python 3.11. A fresh isolated environment created
from the lock must pass `pip check`, the complete test suite, the baseline
rollout, the modular simulation script, and the MPC sanity module. Docker must
be built when a Docker daemon is available; lack of a daemon must be reported,
not treated as a successful build.

This phase changes packaging and documentation only. It must not change
simulation, controller, reward, training, or frontend behavior.
