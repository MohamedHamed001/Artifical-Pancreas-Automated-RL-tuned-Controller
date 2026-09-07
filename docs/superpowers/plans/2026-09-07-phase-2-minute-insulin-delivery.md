# Phase 2A Minute-Resolved Insulin Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply every meal bolus and split-bolus tail at its exact simulation minute, independent of the outer controller interval.

**Architecture:** The controller continues to propose a basal rate once per `SimulationRunner.step()`. The runner applies that fixed basal proposal, meal events, tail delivery, and the existing safety gate once for every simulated minute. `InsulinCalculator` owns dose calculation and its tail queue; it does not own simulation timing. `StepRecord` fields remain interval-average rates in U/h for compatibility.

**Tech Stack:** Python 3.11, NumPy, pytest.

**Spec:** `CodeReview/plan.md` Phase 2; `docs/superpowers/plans/2026-09-05-phase-1-stabilization.md` scenario acceptance blocker.

## Global Constraints

- Preserve controller decision cadence and existing public runner/configuration APIs.
- Treat `immediate_dose` as units delivered across the event minute: convert it once to `U/h` with `dose_u * 60`.
- Call `drain_tail_dose()` exactly once per simulated minute; it already returns `U/h` for that minute.
- Keep safety mandatory for every minute-level total rate, and do not make clinical-validity claims.
- Preserve the generic raw scenario-array contract and Phase 1 structured-event fixes.
- Do not change Hovorka equations, safety policy thresholds, IOB physiology, frontend, controller code, or checkpoint behavior.
- Do not commit or push.

### Task 1: Lock the insulin time and dose contract

**Files:**
- Modify: `src/ap_rl/utils/insulin_calculator.py`
- Test: `tests/test_insulin_calculator.py`

**Interfaces:**
- Consumes: `deliver_bolus()` returning `immediate_dose` and queueing `tail_dose`; `drain_tail_dose()` returning a one-minute `U/h` rate.
- Produces: a first meal at minute zero that is eligible for dosing and a documented one-minute accounting contract.

- [x] Add a failing test that `set_current_time(0); deliver_bolus(...)` succeeds for a new calculator.
- [x] Change the initial and reset `last_insulin_time` to negative infinity. Keep the configured lockout behavior for later meal events.
- [x] Add a test that summing `drain_tail_dose()/60` until exhaustion equals the returned `tail_dose` within floating-point tolerance.
- [x] Run the calculator tests.

### Task 2: Apply commands at minute resolution

**Files:**
- Modify: `src/ap_rl/simulation/simulator.py`
- Test: `tests/unit/test_scenario_inputs.py`

**Interfaces:**
- Consumes: one controller basal proposal in U/h, zero or more gram meal events per minute, calculator immediate/tail doses, and `SafetySupervisor.evaluate()`.
- Produces: one patient step and one safety evaluation per minute; an outer `StepRecord` containing arithmetic-mean requested, delivered, and bolus rates across its interval.

- [x] Add a failing `dt_min=5` case-10 test that records `deliver_bolus()` calls and proves the 78.06 g event at minute 172 is evaluated exactly once.
- [x] Add a failing test that a known bolus contributes its complete immediate dose in the event minute, irrespective of `dt_min`.
- [x] Move meal lookup, calculator time update, immediate-rate conversion, tail drain, total command construction, safety evaluation, and patient stepping into the existing minute loop. The controller action remains outside that loop.
- [x] Accumulate minute rates and return their arithmetic mean in the outer record. De-duplicate safety event names while preserving first-occurrence order.
- [x] Run focused scenario tests and the existing integration tests that exercise the runner.

### Task 3: Verify time-step invariance

**Files:**
- Test: `tests/unit/test_scenario_inputs.py`

**Interfaces:**
- Consumes: the minute-resolved runner from Task 2.
- Produces: a regression guard that the same event has the same integrated requested non-basal dose for `dt_min=1` and `dt_min=5`.

- [x] Add a test that captures minute requested rates for a single known meal under both step sizes and compares `sum(rate_u_h / 60)`.
- [x] Add a reset test proving a tail queued in one episode does not appear after `SimulationRunner.reset()`.
- [x] Run the full fresh-environment suite (68 tests), the three Phase 0 smoke commands, `uv lock --check`, and `git diff --check`.

## Explicit Deferrals

- Replacing the existing safety policy with configurable total-command limits and physiologically validated IOB belongs to the remainder of Phase 2.
- How a physical pump quantizes or models a true instantaneous bolus is not represented by this minute-resolution model.
- The canonical Hovorka equation and exercise-model decisions remain outside this delivery-accounting task.
