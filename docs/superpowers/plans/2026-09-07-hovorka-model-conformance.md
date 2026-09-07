# Hovorka Model Conformance and Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a testable Hovorka 2004 base model, then layer the project's circadian and exercise innovations on it without unit ambiguity.

**Architecture:** Keep `hovorka_ode` as the sole pure RHS, `HovorkaPatientModel` as the owner of mutable state and exercise edge tracking, and `pack_params` as the one boundary that scales per-kilogram parameters. The canonical base is selected by configuration; the project digital-twin configuration adds circadian and exercise parameters without creating a second physics implementation.

**Tech Stack:** Python 3.11, NumPy, SciPy reference integration, pytest, optional Numba.

**Spec:** `docs/superpowers/specs/2026-09-07-hovorka-model-contract.md`

## Global Constraints

- Time is minutes; internal insulin concentration is U/L; public delivery rates are U/h.
- The 2004 paper is the base-model source. Circadian and exercise are project extensions, not paper features.
- Preserve the current values in a complete named legacy profile before replacing defaults.
- Do not retrain, evaluate, or reuse current RL checkpoints as evidence after physics changes.
- Do not commit or push without the user's explicit approval in that turn.
- Do not change controller, reward, safety, frontend, or public step signatures in this plan.

## Phase Map

| Phase | Deliverable | Gate before the next phase |
|---|---|---|
| 0 | Legacy snapshot and executable unit tests | Current behaviour is reproducible and labelled legacy. |
| 1 | Exact base-Hovorka RHS and canonical profile | Equation-level tests pass with extensions disabled. |
| 2 | Circadian EGP extension | Zero amplitude is base-identical; phase/amplitude tests pass. |
| 3 | Exercise sensitivity extension | Original `K_rise` unit is confirmed and edge/RHS tests pass. |
| 4 | Scientific regression suite and downstream reset | Reference tests pass; checkpoints are marked obsolete. |
| 5 | Controller, API, and custom-frontend work | Separate plans only after Phase 4 acceptance. |

### Task 1: Freeze the legacy numerical contract (Phase 0)

**Files:**
- Create: `configs/profiles/legacy_pre_conformance.yaml`
- Create: `tests/unit/test_hovorka_units.py`
- Modify: `src/ap_rl/envs/defaults.py`
- Modify: `configs/patient_default.yaml`

**Interfaces:**
- Consumes: `DEFAULT_PATIENT_PARAMS: dict` and `pack_params(params, body_weight)`.
- Produces: a named, complete legacy profile and direct tests for runtime units.

- [ ] **Step 1: Write the failing unit-conversion test**

```python
def test_runtime_kb_values_are_for_u_per_litre() -> None:
    params = DEFAULT_PATIENT_PARAMS
    packed = pack_params(params, body_weight=75.0)
    assert packed[3:6] == pytest.approx([0.03072, 0.0492, 1.56])
```

- [ ] **Step 2: Run it to verify the old defaults fail**

Run: `uv run pytest tests/unit/test_hovorka_units.py::test_runtime_kb_values_are_for_u_per_litre -q`

Expected: FAIL because the active values are `[0.003, 0.06, 0.04]`.

- [ ] **Step 3: Snapshot the current parameters before changing defaults**

Copy every key/value from the current default parameter dictionary into
`legacy_pre_conformance.yaml`, including circadian and exercise fields. Label
it synthetic legacy behaviour; do not describe it as paper-conformant.

- [ ] **Step 4: Add the canonical base parameter dictionary**

Make `DEFAULT_PATIENT_PARAMS` the extension-disabled canonical base. Keep
`A_EGP: 0.0` and `F_peak: 1.0`; use the runtime `k_b` values from the spec.
Keep the dictionary flat and explicit rather than adding a profile factory.

- [ ] **Step 5: Run the focused test and profile-loader tests**

Run: `uv run pytest tests/unit/test_hovorka_units.py tests/test_app_patient_config.py -q`

Expected: PASS.

### Task 2: Conform the base RHS to Hovorka 2004 (Phase 1)

**Files:**
- Modify: `src/ap_rl/simulation/hovorka.py:16-112`
- Modify: `src/ap_rl/simulation/patient.py:48-89`
- Modify: `configs/patient_default.yaml`
- Create: `tests/unit/test_hovorka_base_equations.py`
- Modify: `tests/regression/test_golden_trajectories.py`

**Interfaces:**
- Consumes: `hovorka_ode(t, y, u_i_min, u_g_g_min, f_sens, p)`; the external
  glucose input is grams per minute.
- Produces: a base RHS whose `f_sens=1` and `A_EGP=0` path implements equations (1)-(7).

- [ ] **Step 1: Write equation-level failing tests**

Use a non-zero state and calculate the expected terms directly from the paper:

```python
expected_dq1 = (
    u_id + egp - renal - f01_corrected - x1 * q1 + k12 * q2
)
expected_dq2 = x1 * q1 - (k12 + x2) * q2
assert derivative[6] == pytest.approx(expected_dq1)
assert derivative[7] == pytest.approx(expected_dq2)
```

Add separate cases for `G=4.0 mmol/L` and `G=5.0 mmol/L` to prove the
`F_01 * G / 4.5` correction occurs only below the threshold.

- [ ] **Step 2: Run the equation tests against the old RHS**

Run: `uv run pytest tests/unit/test_hovorka_base_equations.py -q`

Expected: FAIL because the current Q1/Q2 transfer terms and low-glucose path
differ from the source equations.

- [ ] **Step 3: Implement only the published base terms**

Replace the glucose equations with:

```python
dQ1 = u_id + egp - f_r - f_01_total - x1 * Q1 + k_12 * Q2
dQ2 = x1 * Q1 - (k_12 + x2) * Q2
```

Restore the documented `F_01` branch. In `patient.py`, initialize `k_a3` and
the effective `k_b` values from the same canonical dictionary used by packing;
do not duplicate fallback numbers.

- [ ] **Step 4: Update the canonical YAML and rebase golden tests**

Use the canonical values from the spec, not the legacy values. Replace old
golden numbers only after recording a deterministic trace generated by the
paper-conformant, extension-disabled profile.

- [ ] **Step 5: Verify base-model numerical and regression contracts**

Run: `uv run pytest tests/unit/test_hovorka_base_equations.py tests/unit/test_hovorka_integrator.py tests/regression/test_golden_trajectories.py -q`

Expected: PASS.

### Task 3: Make circadian EGP an isolated project extension (Phase 2)

**Files:**
- Modify: `src/ap_rl/simulation/hovorka.py:57-60`
- Create: `configs/profiles/project_circadian.yaml`
- Create: `tests/unit/test_circadian_egp.py`

**Interfaces:**
- Consumes: `A_EGP: float` and `phi_EGP: float` in minutes.
- Produces: an EGP multiplier applied before `(1 - x3)` insulin suppression.

- [ ] **Step 1: Write the failing extension tests**

```python
assert circadian_multiplier(0.0, amplitude=0.0, phase_min=-60.0) == 1.0
assert circadian_multiplier(300.0, amplitude=0.05, phase_min=-60.0) == pytest.approx(1.05)
assert circadian_multiplier(1020.0, amplitude=0.05, phase_min=-60.0) == pytest.approx(0.95)
```

The tests must also compare complete RHS output at `A_EGP=0` with the base
RHS output, proving the extension is inert when disabled.

- [ ] **Step 2: Run the tests and confirm the requested seam is missing or private**

Run: `uv run pytest tests/unit/test_circadian_egp.py -q`

Expected: FAIL until the small multiplier helper is exposed for direct testing.

- [ ] **Step 3: Extract only the pure multiplier**

Add a small module-level function usable by both the RHS and tests; retain the
existing formula exactly:

```python
return 1.0 + amplitude * np.sin(2.0 * np.pi * (time_min - phase_min) / 1440.0)
```

The Numba-decorated RHS must call a Numba-compatible implementation or retain
the expression inline; do not add a class or clock subsystem.

- [ ] **Step 4: Add the project circadian profile**

Set `A_EGP: 0.05` and `phi_EGP: -60` only in the extension profile. The base
default remains amplitude zero.

- [ ] **Step 5: Verify extension isolation**

Run: `uv run pytest tests/unit/test_circadian_egp.py tests/unit/test_hovorka_base_equations.py -q`

Expected: PASS.

### Task 4: Connect the exercise sensitivity extension (Phase 3)

**Files:**
- Modify: `src/ap_rl/simulation/hovorka.py:48-52`
- Modify: `src/ap_rl/simulation/patient.py:143-163`
- Create: `configs/profiles/project_exercise.yaml`
- Create: `tests/unit/test_exercise_sensitivity.py`

**Interfaces:**
- Consumes: `exercise_active: bool`, `F_peak: float`, `K_rise: float`, `K_decay: float`.
- Produces: dimensionless `f_sens` supplied to `hovorka_ode` and multiplied
  into each `k_bi * I` activation term.

- [ ] **Step 1: Record the exercise-rate calibration assumption**

The source documents do not state a numeric unit for the remembered literal
`K_rise = 5`. The implementation ruling is `5 h^-1 = 5 / 60 min^-1`; record
that explicit assumption in `project_exercise.yaml` beside the minute-based
value so a recovered Amesim source can replace it without touching the RHS.

- [ ] **Step 2: Write failing edge and RHS tests**

```python
assert f_at_start == pytest.approx(1.0)
assert f_at_end == pytest.approx(f_peak)
assert f_after_end == pytest.approx(f_peak)
assert f_long_after_end == pytest.approx(1.0, abs=1e-6)
assert dx_with_exercise == pytest.approx(
    f_sens * k_b1 * insulin - k_a1 * x1
)
```

Repeat the final assertion for `x2` and `x3`; exercise must alter all three
activation paths and no other base term directly.

- [x] **Step 3: Record the initial disconnected-extension failure**

Run: `uv run pytest tests/unit/test_exercise_sensitivity.py -q`

Initial RED: FAIL because `f_sens` was ignored by the RHS before Step 4.

- [ ] **Step 4: Apply the one-line RHS coupling**

```python
dx1 = f_sens * k_b1 * I - k_a1 * x1
dx2 = f_sens * k_b2 * I - k_a2 * x2
dx3 = f_sens * k_b3 * I - k_a3 * x3
```

Keep the existing rising/falling-edge state in `HovorkaPatientModel`; do not
add exercise effects directly to EGP or glucose disposal.

- [ ] **Step 5: Verify continuity and base parity**

Run: `uv run pytest tests/unit/test_exercise_sensitivity.py tests/unit/test_hovorka_integrator.py -q`

Expected: PASS, including a test that `F_peak=1` makes an exercised run equal
to a non-exercised run.

### Task 5: Rebaseline scientific verification (Phase 4)

**Files:**
- Modify: `tests/regression/test_golden_trajectories.py`
- Create: `tests/integration/test_hovorka_extension_profiles.py`
- Modify: `README.md`
- Modify: `docs/phase-0-baseline.md`

**Interfaces:**
- Consumes: canonical base, legacy, circadian, and exercise profiles.
- Produces: repeatable profile traces and explicit limits of interpretation.

- [ ] **Step 1: Add deterministic profile-comparison tests**

Verify base equals circadian with `A_EGP=0`, base equals exercise with
`F_peak=1`, and a non-trivial exercise session changes `x1`, `x2`, and `x3`
in the expected direction without NaN, Inf, or negative stored masses.

- [ ] **Step 2: Re-run RK4-versus-RK45 checks for every model layer**

Run: `uv run pytest tests/unit/test_hovorka_integrator.py tests/integration/test_hovorka_extension_profiles.py -q`

Expected: PASS with the existing numerical tolerance; this is an integration
accuracy check, not a clinical-validity claim.

- [ ] **Step 3: Mark checkpoint provenance invalid for the new physics**

Update documentation to state that existing checkpoints were trained on the
legacy physics and must not be used to compare the conformance model.

- [ ] **Step 4: Run the complete scientific-core gate**

Run:

```bash
uv lock --check
uv run pytest tests/unit tests/regression tests/integration -q
uv run python scripts/smoke_baseline_rollout.py
uv run python scripts/verify_modular_sim.py
git diff --check
```

Expected: PASS, except any explicitly documented external runtime limitation.

### Phase 5: Downstream work after the scientific-core gate

Create separate implementation plans only after Phase 4 is accepted:

1. Rework controller safety and IOB against the conformance model.
2. Recreate evaluation datasets and train new controller/RL checkpoints.
3. Define a narrow simulation-service API.
4. Build the custom frontend against that API; do not revive Streamlit.

## Self-review

- **Spec coverage:** Phase 0 preserves current behaviour, Phase 1 covers the
  base equations and units, Phases 2-3 isolate both project innovations, and
  Phase 4 prevents invalid reuse of old results.
- **Scientific boundary:** this plan does not claim the extensions are part of
  Hovorka 2004 or that passing numerical tests establishes clinical validity.
- **Open decision:** only the original numeric unit for `K_rise = 5` blocks
  Phase 3. All other phases have a declared source and acceptance gate.
