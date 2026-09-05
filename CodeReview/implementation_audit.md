# Architecture Audit - 2026-05-15 (Recheck)

Audit basis:

- target claims in `final_architecture.md`
- current working tree under `src/`, `app/`, `scripts/`, `tests/`
- execution checks:
  - `pytest`
  - `python scripts/smoke_baseline_rollout.py`
  - `python scripts/verify_modular_sim.py`
  - `python scripts/parity_check.py`
  - direct modular run via `runtime.run_simulation`

## Current Verdict

The previous audit needed revision because the working tree has moved forward. The modular architecture is more complete than before, and the core execution path is now functional enough for tests and smoke rollouts. However, `final_architecture.md` still overstates readiness.

Current high-level conclusion:

- the modular stack is real and actively wired into the runtime
- core tests are now green
- smoke rollout works
- the system is still not production-ready
- parity / architecture-consistency validation is still incomplete
- there is still no backend/API/auth/database layer

## Completed Systems

### 1. Modular simulation core

Verified:

- `src/ap_rl/simulation/patient.py` defines `PatientModel` and `HovorkaPatientModel`
- `src/ap_rl/simulation/hovorka.py` contains RK4-based Hovorka stepping
- `src/ap_rl/simulation/integrators.py` provides reusable RK4 stepping
- `src/ap_rl/simulation/simulator.py` defines `SimulationConfig` and `SimulationRunner`

This is substantive implementation, not placeholder structure.

### 2. Typed core domain model

Verified:

- `src/ap_rl/core/types.py` defines `PatientState`, `InsulinCommand`, `SafetyDecision`, `PatientConfig`, and `Scenario`
- `src/ap_rl/core/records.py` defines the runtime-facing `StepRecord` and `EpisodeRecord`

### 3. Safety layer in the modular path

Verified:

- `src/ap_rl/controllers/safety.py` implements the active modular `SafetySupervisor`
- `SimulationRunner` uses it directly

### 4. Observation-builder path

Verified:

- `src/ap_rl/simulation/observations.py` defines `ObservationBuilder`
- `SimulationRunner` and `DiabetesPIDEnv` use it for the 19-D observation path

### 5. Scenario loading for active workflows

Verified:

- `src/ap_rl/utils/scenarios.py` now implements `load_case()`
- `DiabetesPIDEnv` uses that loader successfully
- smoke rollout now initializes and runs

### 6. Test and smoke baseline health

Observed:

- `pytest`: `43 passed`
- `python scripts/smoke_baseline_rollout.py`: succeeds

This is a major improvement over the prior broken state.

## Partial Systems

### 1. `DiabetesPIDEnv` as a wrapper over `SimulationRunner`

This is now mostly true.

Verified:

- `src/ap_rl/envs/diabetes_pid_env.py` delegates stepping through `self.runner.step(...)`
- it is now a `gymnasium.Env`
- reset/step contracts are aligned with tests

Still partial because:

- it retains legacy PID-delta tuning semantics
- it still depends on `_skip_reload` in scripts/tests for injected schedules
- it is both a compatibility surface and the active training env

### 2. Frontend integration

The app has partially moved to the modular path.

Verified:

- `app/app.py` imports `run_simulation`, `PIDController`, `SupervisoryController`, and `PatientConfig`
- the app no longer appears to depend on `_skip_reload`

Still partial because:

- `_build_patient_config()` is incompatible with the current `PatientConfig` dataclass
  - app uses `weight_kg` and `seed`
  - actual dataclass requires `name`, `params`, `body_weight_kg`
- `app/app.py` still imports `DiabetesPIDEnv` even though the visible episode runner uses `run_simulation`
- there is no backend boundary; the app is still an in-process UI

### 3. Training integration

Verified:

- training still uses `ap_rl.envs.DiabetesPIDEnv`
- A2C remains the active training path

Still partial because:

- training is not fully re-centered on the modular abstractions
- no trainer abstraction, callback layer, or experiment service has been added

### 4. Verification scripts

Verified:

- `scripts/verify_modular_sim.py` runs successfully
- `scripts/parity_check.py` exists and exercises parity logic

Still partial because:

- parity validation is not passing
- the parity script currently fails due to scenario-data format mismatch

## Missing Systems

### Backend / API layer

Missing:

- no service backend
- no request/response schema layer for an external API
- no streaming or playback service boundary

### Database / schema persistence

Missing:

- no database
- no migrations
- no ORM or repository layer

### Authentication / authorization

Missing:

- no auth model
- no user/session boundary
- no authorization flow

### Production service concerns

Missing:

- no health checks
- no service deployment contract
- no production process topology beyond Streamlit/scripts

## Risky or Fragile Areas

### 1. App config typing is still wrong

Evidence:

- `_build_patient_config()` in `app/app.py` constructs `PatientConfig` with fields that do not match the current dataclass

Impact:

- the app path is still at risk of runtime failure when that function is exercised under the current type contract

### 2. Parity path is still broken

Observed:

- `python scripts/parity_check.py` fails

Current failure:

- `SimulationRunner._get_scenario_meal()` expects list entries to be dict-like
- `parity_check.py` passes `baseline["meals"]`, whose current structure does not match that expectation in the failing case

Impact:

- the “zero-drift architecture” claim is not validated

### 3. Duplicate architecture surfaces remain

Examples:

- `src/ap_rl/core/types.py` vs `src/ap_rl/core/records.py`
- modular simulation path vs legacy-compatible patient wrapper in `src/ap_rl/simulation/legacy_patient.py`
- `rewards/standard.py` and `rewards/clinical.py`

Impact:

- the system works, but the authoritative architectural boundary is still not singular

### 4. Reward architecture is still mixed

Observed:

- modular runner uses `src/ap_rl/rewards/standard.py::ClinicalZoneReward`
- that reward still includes catastrophic penalties like `-10000`
- separate `src/ap_rl/rewards/clinical.py` also exists

Impact:

- reward design is still split between old and new approaches
- architecture intent is not fully normalized

### 5. Legacy control semantics still dominate

Observed:

- `DiabetesPIDEnv` remains the active training surface
- action is still PID gain delta tuning

Impact:

- the control architecture is still aligned with the old meta-control design, not a fully direct-command or hierarchical supervisory design

## Architectural Deviations from `final_architecture.md`

### Claim: “production-ready modular architecture is complete”

Assessment:

- false

Reason:

- no backend/API/auth/database/service layer
- no production hardening
- parity validation still broken
- app/runtime contracts still have mismatches

### Claim: `DiabetesPIDEnv` is now a thin wrapper around `SimulationRunner`

Assessment:

- mostly true, but overstated

Reason:

- step execution is delegated to `SimulationRunner`
- but the env still carries compatibility behavior and legacy RL action semantics

### Claim: “Zero-Drift Architecture”

Assessment:

- not established

Reason:

- parity script does not currently produce a clean parity result

### Claim: verification confirms stability

Assessment:

- partially true

Reason:

- smoke rollout works
- tests pass
- modular verification run works
- parity verification still fails

## Recommended Next Priorities

1. Fix `app/app.py`’s `PatientConfig` construction so the frontend matches the live core datamodel.
2. Fix `scripts/parity_check.py` and normalize scenario/meals input handling so parity becomes a real validation tool.
3. Unify reward usage so the modular runner and clinical reward design are not split across `standard.py` and `clinical.py`.
4. Decide whether `DiabetesPIDEnv` remains the primary RL env or becomes a compatibility layer; right now it is both.
5. Remove remaining duplicate architecture surfaces where possible:
   - consolidate record contracts
   - reduce compatibility-only layers that are no longer needed
6. Keep `final_architecture.md` honest:
   - modularized: yes
   - production-ready/complete: not yet
