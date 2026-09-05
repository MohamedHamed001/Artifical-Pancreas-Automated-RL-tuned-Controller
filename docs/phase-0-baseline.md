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

The following commands execute successfully on the baseline working tree:

```bash
python -m pytest -p no:cacheprovider -q
python scripts/smoke_baseline_rollout.py
python scripts/verify_modular_sim.py
PYTHONPATH=src python -m ap_rl.scripts.sanity_check_sim
```

These checks show that the code imports and runs. They do not establish
physiological correctness, numerical parity, controller safety, or clinical
validity.

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
- Existing checkpoints have mixed observation dimensions and lack provenance.

These issues are intentionally not hidden by this baseline. They should be
addressed test-first in the scientific-contract, simulator, and RL phases.

## Local/generated artifacts

Generated baseline JSON, simulation plots, local status files, assistant
settings, model checkpoints, and the `scratch/` directory are excluded from
version control. Reproducible source, tests, configuration, and diagnostic
scripts remain publishable.
