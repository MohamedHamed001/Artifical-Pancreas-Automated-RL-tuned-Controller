# Phase 1 Core Simulation Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Hovorka runtime deterministic, explicit about inputs, and testable without silently changing its current physiology.

**Architecture:** Keep `hovorka_ode`, `rk4_step`, and `HovorkaPatientModel` as the existing deep module seam. The patient owns mutable state and event application; the RHS remains pure; `legacy_patient.py` remains the only compatibility adapter. Do not add a second physics implementation.

**Tech Stack:** Python 3.11, NumPy, SciPy (RK45 reference tests only), pytest, optional Numba.

**Spec:** `CodeReview/plan.md` — Phase 1, with the evidence and constraints below.

## Global Constraints

- Preserve the currently explicit parameter values; remove hidden fallback divergence, but do not reset every `t_max_G` to 30 or 40.
- Preserve public `HovorkaPatientModel.step()` and `legacy_patient.HovorkaPatient` compatibility semantics until a timestamp decision is approved.
- Do not claim clinical correctness; separate numerical stabilization from equation or exercise-model changes.
- Do not commit or push as part of implementation.

## Current Status and Evidence

- The active paths already converge on `src/ap_rl/simulation/hovorka.py` through `patient.py`; the old `envs/hovorka_patient.py` is gone.
- The active `hovorka_ode` now applies `f_sens`, names its optional external
  glucose input `u_g_g_min`, and follows the accepted Q-compartment equations.
- `TestCaseManager/t1dm_mgr_generator.py:246-287` writes legacy meal arrays as repeated `mmol/kg` values over a 40-minute window. Raw arrays therefore have no safe generic interpretation at the simulator seam; prefer structured `TestCases.txt` events, while allowing for rounded metadata.
- The published reference is [Hovorka et al. 2004](https://www.stat.yale.edu/~jtc5/diabetes/NonlinearModelPredictiveControl_Hovorka_04.pdf); metadata is also available from [PubMed](https://pubmed.ncbi.nlm.nih.gov/15382830/).
- Phase 0 established the runtime and existing suite: a fresh Python 3.11.15 environment passed lock/install/pip checks, 44 tests, and the three smoke commands. Docker remains unverified because the local daemon was unavailable.
- Phase 1 numerical increment completed: `patient.py` clears `_last_glucose_rate` on reset; seven integrator tests (analytic coupled-RK4, five nominal 24-hour RK45 comparisons, and reset regression) pass. The full suite is 51 passed. Worst current RK4/RK45 glucose difference is approximately `6.87e-7 mg/dL`; this validates numerical integration, not physiological equations. Independent review is pending.
- The numerical/reset increment has been independently reviewed clean (`max`/`RMSE < 1e-3 mg/dL` for the numerical gate). Phase 2A resolved the scenario acceptance blocker: structured case-10 meal 172 now triggers its bolus at minute 172 even when `dt_min=5`, without changing controller cadence. Its 68-test fresh-environment verification passed; the broader equation, parameter-validation, and exercise decisions remain open.

## File Scope

- Modify: `src/ap_rl/simulation/patient.py`, `src/ap_rl/simulation/hovorka.py`, `src/ap_rl/simulation/simulator.py`, `src/ap_rl/utils/scenarios.py`, and only the compatibility code needed in `src/ap_rl/simulation/legacy_patient.py`.
- Test: `tests/unit/test_hovorka_integrator.py`, `tests/unit/test_hovorka_events.py`, and existing `tests/regression/test_golden_trajectories.py`.
- Do not touch the frontend, controllers, rewards, training, or IOB design in this phase.

## Task 1: Lock current state and numerical contracts

- [x] Add a regression test proving `reset()` clears `_last_glucose_rate`; implement that smallest fix in `patient.py`.
- [ ] Add tests for state order, finite output, U/h-to-U/min conversion, and non-negative final state after each 0.1-minute `hovorka_step` call.
- [ ] Record the parameter-validation design at the existing configuration seam, retaining explicit profile values including `t_max_G`; do not enforce missing-value errors until the partial config callers are reconciled.
- [x] Rename the optional input `u_g_g_min` to state its grams-per-minute boundary contract.
- [ ] Run the focused tests before continuing.

## Task 2: Normalize only unambiguous scenario inputs

- [x] Use structured meal events (`time`, `carbs` in grams) and exercise intervals as the canonical simulator input.
- [x] Prefer bundled `TestCases.txt` structured events. Reject or leave unsupported ambiguous raw `.data` arrays at the loader seam rather than guessing their units or reconstructing overlapping meals; treat rounded metadata as provenance, not exact numerical equivalence.
- [x] Define meal application once at the integer interval start and exercise as a documented interval; preserve current public timestamp semantics until approved.
- [x] Add tests for case-10 minute-172 one-shot meals, coincident meals, exercise start/end, generic-array compatibility, reset, and missing metadata.
- [ ] Keep `legacy_patient.py` as an adapter; do not duplicate event logic.

## Task 3: Verify before changing physiology

- [x] Run the test-only RK4-versus-RK45 reference comparison owned by the parallel test task, using the current equations and identical event split points.
- [x] Compare fasting, meal, insulin, meal-plus-insulin, and circadian cases;
  the later conformance phase adds an exercise rise/stop/decay comparison.
- [ ] Defer any golden-trajectory changes until the scientific model choice is accepted; they are not routine stabilization work.
- [ ] Review the Q1/Q2 equation difference, `Fc01` low-glucose correction, `AG`, `ka3`, and exercise sensitivity as explicit scientific choices—not routine stabilization.

## Verification and Handoff

```bash
uv lock --check
uv run pytest -p no:cacheprovider -q
uv run pytest tests/unit tests/regression/test_golden_trajectories.py -q
uv run python scripts/smoke_baseline_rollout.py
uv run python scripts/verify_modular_sim.py
PYTHONPATH=src uv run python -m ap_rl.scripts.sanity_check_sim
git diff --check
```

The model contract resolved the equation and project-specific exercise choices.
Public timestamps retain interval-start semantics.
