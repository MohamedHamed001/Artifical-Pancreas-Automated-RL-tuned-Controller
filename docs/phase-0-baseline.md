# Phase 0 Branch Baseline

This branch captures the in-progress modular-architecture refactor so that later
work has a reviewable and reproducible starting point. It is a development
baseline, not a production or clinical release.

## Product direction

- The Streamlit application is deprecated and will be replaced by a custom
  frontend after the simulator and application-service contracts are stable.
- Streamlit remains temporarily as compatibility code; it is not an active
  target for repair or feature work.
- The simulation package must remain independent of the future HTTP API and
  frontend.

## Mechanically verified paths

The following commands execute successfully in the locked Python 3.11
environment. From a fresh checkout, install that environment with
`uv sync --locked --all-extras`; `uv` reads the exact dependency graph from
`uv.lock` and installs the project plus every optional group:

- `rl`: TensorFlow and `h5py` for reinforcement-learning training/inference.
- `mpc`: CVXPY for model-predictive-control experiments.
- `demo`: Streamlit and Plotly for the temporary compatibility UI.
- `fast`: Numba for optional numerical acceleration.
- `dev`: pytest and pytest-mock for development verification.

Gymnasium is a base dependency because the environment imports it
unconditionally. The verified commands are:

```bash
uv lock --check
uv run pytest -p no:cacheprovider -q
uv run python scripts/smoke_baseline_rollout.py
uv run python scripts/verify_modular_sim.py
PYTHONPATH=src uv run python -m ap_rl.scripts.sanity_check_sim
```

These checks show that the code imports and runs. The later scientific-core
tests add deterministic equation, integration, and profile-regression checks;
neither gate establishes physiological correctness, controller safety, or
clinical validity.

## Checkpoint provenance boundary

All existing and downloadable controller checkpoints were trained on the
legacy pre-conformance simulator physics. They must not be used to compare
controller performance with the conformance model. New evaluation data and
checkpoints require separate downstream controller and training work after the
scientific-core gate is accepted.

## Known blockers

- Exercise input reaches the Hovorka function but does not affect its equations.
- Normalized RL observations are passed to controllers that expect physical
  glucose values.
- Meal-event, meal-rate, bolus, IOB, and COB unit contracts are inconsistent.
- Bolus delivery depends incorrectly on the simulation step duration.
- Explicit carb-ratio, insulin-sensitivity, and scenario-root configuration is
  not propagated consistently.
- The primary Gymnasium environment does not reset PID gains deterministically.
- The all-profiles trainer, public agent evaluation method, and parity script
  use outdated or incompatible contracts.
- Existing checkpoints have mixed observation dimensions and were trained on
  legacy physics, so they are invalid for conformance-model comparisons.

These issues are intentionally not hidden by this baseline. They should be
addressed test-first in the scientific-contract, simulator, and RL phases.

## Local/generated artifacts

Generated baseline JSON, simulation plots, local status files, assistant
settings, model checkpoints, and the `scratch/` directory are excluded from
version control. Reproducible source, tests, configuration, and diagnostic
scripts remain publishable.
