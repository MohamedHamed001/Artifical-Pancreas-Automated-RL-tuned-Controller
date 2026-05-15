# Artificial Pancreas RL Controller Implementation Roadmap

Source material: every file in `CodeReview/` was read and synthesized:

- `CodeReview/AUDIT_REPORT.md`
- `CodeReview/REFACTOR_BLUEPRINT.md`
- `CodeReview/FILE_REVIEW.md`
- `CodeReview/issues.json`
- `CodeReview/RESEARCH_DIRECTIONS.md`

This roadmap is intentionally direct. The current system is useful as a prototype and demonstration scaffold, but it is not a reliable research platform yet. The safest path is a staged partial rewrite: preserve scientifically useful and well-structured modules, delete or rebuild unsafe architectural boundaries, and postpone UI polish until simulator, safety, control, evaluation, and reproducibility are stable.

This project must remain framed as a research and engineering simulator. It is not a clinical product, not a medical device, and not clinically validated.

---

# 1. Executive Summary

## Current System State

The repository currently implements an artificial pancreas simulator centered on `src/ap_rl/envs/diabetes_pid_env.py`. That environment combines a Hovorka virtual patient, a PID controller, an A2C actor/critic, meal and exercise schedules, bolus logic, reward computation, safety clamps, state construction, file I/O, rollout statistics, and rendering-oriented history in one class.

The system can run 24-hour simulations and produce demo plots. It has some strong pieces:

- The Hovorka patient model equations in `src/ap_rl/envs/hovorka_patient.py` are a reasonable basis for simulation.
- `src/ap_rl/utils/insulin_calculator.py` contains useful clinical heuristics such as the 500 rule, 1500 rule, correction dead-band, lockout, and split/tail bolus handling.
- `src/ap_rl/envs/profile_loader.py`, `src/ap_rl/envs/scenario_builder.py`, and `src/ap_rl/utils/config.py` are simple and worth preserving.
- `src/ap_rl/runtime/rollout.py` has a useful `EpisodeRecord` concept.
- `src/ap_rl/evaluation/metrics.py` and `src/ap_rl/visualization/publication.py` are clean foundations, though incomplete.

However, the current architecture is not safe or stable enough to support credible RL/control results. The main system behavior is dominated by avoidable design problems, not by the underlying control research.

## Main Architectural Problems

The most important problem is the RL formulation. The A2C agent outputs incremental PID gain changes `(dKp, dKi, dKd)` instead of an insulin command or a high-level supervisory decision. This creates a meta-control problem:

1. RL adjusts PID gains.
2. PID transforms glucose error into a PID output.
3. PID output is scaled by `0.01`.
4. That scaled value adjusts basal insulin.
5. Insulin affects glucose only after a long pharmacokinetic delay.

This makes credit assignment unnecessarily hard and creates gain drift, non-stationarity, and unsafe oscillation. The agent is learning to tune the steering sensitivity while the vehicle is moving instead of learning how to steer or supervising a controller that already knows how to steer.

The second major problem is simulator inconsistency. `src/ap_rl/envs/hovorka_patient.py` uses:

- Python fallback: SciPy `solve_ivp` with RK45 and circadian EGP modulation.
- Numba path: one-minute forward Euler and no circadian EGP modulation.

These are different physics. A policy trained with one path can silently evaluate on another. This blocks meaningful claims about learning, generalization, or controller quality.

The third major problem is reward design. The reward in `DiabetesPIDEnv` mixes a peaked target near 100 mg/dL, PID target 120 mg/dL, rate penalties, IOB brake terms, recovery bonuses, stability bonuses, and catastrophic penalties such as `-10000`. The review material reports training episodes where TIR is excellent but reward is extremely negative. That means the optimizer is not optimizing the clinical objective.

The fourth major problem is safety. Current safety logic clamps basal insulin but does not reliably constrain total insulin after bolus delivery. IOB is tracked by a simplistic exponential decay, not by the insulin pharmacokinetics already present in the Hovorka states. There is no complete command-level safety supervisor that every controller must pass through.

The fifth major problem is system structure. The environment is a god class, training logic is duplicated and monkey-patched, TensorFlow imports occur too early, the UI reaches into environment internals, and temp files are written on every reset. These issues block parallel training, reproducibility, and maintainability.

## Biggest Technical Risks

The risks below must be treated as blockers before further RL claims.

| Risk | Why it matters | Required response |
|---|---|---|
| Numba/Python Hovorka divergence | Different physics depending on installed packages | Replace with one deterministic integrator and validate against RK45 reference |
| PID-delta RL action space | Poor credit assignment, gain drift, unsafe non-stationarity | Deprecate as primary approach; replace with direct insulin env and hybrid MPC+RL supervisor |
| Reward/TIR mismatch | Training objective conflicts with clinical quality | Replace with bounded clinical reward and select checkpoints by clinical metrics |
| Incomplete insulin safety | Bolus and total insulin can bypass basal-only clamp | Introduce mandatory `SafetySupervisor` over full insulin commands |
| Wrong IOB accounting | Exponential decay underestimates active insulin | Compute IOB from pharmacokinetic compartments or a validated two-compartment model |
| Temp file reset loop | Breaks parallel envs and reproducibility | Pass scenario data in memory |
| No golden trajectories | Silent simulator regressions are undetectable | Add deterministic replay and golden trajectory tests |
| UI/backend coupling | Demo reruns expensive simulations and reaches into internals | Split simulation service from UI after core correctness |

## Recommended Overall Direction

The correct target is a modular research platform:

1. A deterministic, validated simulation core.
2. A clinically conservative safety supervisor.
3. Multiple interchangeable controllers.
4. MPC as the primary controller baseline.
5. RL demoted to a slower supervisory/adaptation layer.
6. Standard Gymnasium environments for experiments.
7. Reproducible training and evaluation infrastructure.
8. A thin visualization UI that consumes simulation results through stable APIs.

The target controller hierarchy should be:

```text
RL Supervisor, slow, optional, every 15-30 min
  -> adjusts target, aggressiveness, sensitivity estimate, MPC weights

MPC Controller, primary, every 5 min
  -> computes basal/correction strategy from prediction horizon and constraints

Safety Supervisor, mandatory, every 1 min
  -> clamps, suspends, rejects, or modifies insulin command

Simulation Core
  -> patient, insulin PK, meal absorption, exercise, sensor, pump
```

No learned component should ever bypass the safety supervisor.

## Partial Rewrite or Full Rewrite?

Do not rewrite everything at once. A full rewrite would discard useful validated code and create a large uncontrolled migration. The correct approach is a staged partial rewrite:

Preserve or lightly refactor:

- `src/ap_rl/envs/hovorka_patient.py` equations, after fixing integration and APIs.
- `src/ap_rl/envs/defaults.py`, moved into a clearer constants/config area.
- `src/ap_rl/envs/profile_loader.py`.
- `src/ap_rl/envs/scenario_builder.py`.
- `src/ap_rl/utils/config.py`.
- `src/ap_rl/utils/paths.py`.
- `src/ap_rl/utils/seed.py`, extended for PyTorch.
- `src/ap_rl/utils/insulin_calculator.py`, placed behind safety constraints.
- `src/ap_rl/runtime/rollout.py` concepts, with lazy framework imports.
- `src/ap_rl/evaluation/metrics.py`, expanded.
- `src/ap_rl/visualization/publication.py`, preserved and extended.

Rewrite or delete:

- Rewrite `src/ap_rl/envs/diabetes_pid_env.py` as multiple modules.
- Delete and rebuild `src/ap_rl/envs/hovorka_gym_env.py`.
- Replace `src/ap_rl/utils/pid_controller.py` or rewrite it as a discrete-time PID.
- Refactor/rewrite `src/ap_rl/agents/*` and `src/ap_rl/training/train_a2c.py`.
- Rewrite `app/app.py` after core APIs stabilize.
- Delete `docs/legacy-pid-tuner/`.
- Delete `TestCaseManager/`.
- Delete `verify_best_model.py`.
- Remove temp-file I/O and `atexit` cleanup.

---

# 2. High-Level Target Architecture

## Target Module Boundaries

The target architecture should separate physics, decisions, safety, learning, evaluation, and UI.

```text
src/ap_rl/
  core/
    types.py
    constants.py
    units.py
    config.py
    time.py

  simulation/
    simulator.py
    patient.py
    hovorka.py
    integrators.py
    insulin_pk.py
    meal_model.py
    exercise.py
    sensor.py
    pump.py
    scenario.py

  controllers/
    base.py
    pid.py
    mpc.py
    rl_supervisor.py
    hybrid.py
    oracle.py
    safety.py

  environments/
    glucose_control_env.py
    supervisory_env.py
    wrappers.py

  rewards/
    clinical.py

  training/
    trainer.py
    curriculum.py
    imitation.py
    callbacks.py
    experiment.py

  agents/
    ppo.py
    sac.py
    networks.py
    replay.py

  evaluation/
    metrics.py
    harness.py
    monte_carlo.py
    stress_tests.py
    comparison.py
    golden.py

  visualization/
    glucose_plot.py
    insulin_plot.py
    agp_report.py
    controller_viz.py
    comparison.py

  app/
    server.py
    streamlit_app.py
    schemas.py
    components/
```

## Data Flow

The intended runtime flow is:

```text
ScenarioConfig + PatientConfig + ControllerConfig
  -> ExperimentManager creates seeded run context
  -> ScenarioEngine emits exogenous events
  -> SensorModel reports observed glucose from true patient state
  -> Controller observes ControllerState
  -> Controller proposes InsulinCommand
  -> SafetySupervisor constrains InsulinCommand
  -> PumpModel applies quantization and delivery limits
  -> SimulationCore advances patient state
  -> MetricsEngine receives StepRecord
  -> Logger writes trajectory, config, seed, metrics, and artifacts
```

The controller never mutates the patient directly. It only receives observations and returns commands.

## Timing Flow

Use explicit simulation time everywhere. Do not use wall-clock time in any control code.

Recommended default timing:

- Internal integration step: 15 seconds or 30 seconds.
- Simulation output step: 1 minute.
- CGM observation step: 5 minutes by default, with optional 1-minute ideal sensor mode.
- Safety supervisor step: 1 minute.
- Pump command step: 1 or 5 minutes, configurable.
- MPC decision step: 5 minutes.
- RL supervisory decision step: 15 to 30 minutes.
- Evaluation metrics step: use every available glucose sample but report at CGM-equivalent intervals when comparing to clinical standards.

All modules must take `dt_min` or an explicit timestamp. No module should call `time.time()` for simulation logic.

## Core Interfaces

Define shared dataclasses in `src/ap_rl/core/types.py`.

```python
@dataclass(frozen=True)
class PatientState:
    time_min: int
    glucose_mgdl: float
    glucose_rate_mgdl_min: float
    iob_u: float
    cob_g: float
    exercise_active: bool
    compartments: dict[str, float]

@dataclass(frozen=True)
class SensorReading:
    time_min: int
    glucose_mgdl: float
    is_missing: bool = False
    noise_std_mgdl: float | None = None

@dataclass(frozen=True)
class InsulinCommand:
    time_min: int
    basal_u_h: float
    bolus_u: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SafetyDecision:
    requested: InsulinCommand
    delivered: InsulinCommand
    active_constraints: list[str]
    predicted_min_glucose_mgdl: float | None
    explanation: str

@dataclass(frozen=True)
class StepRecord:
    time_min: int
    true_state: PatientState
    sensor: SensorReading
    requested_command: InsulinCommand
    delivered_command: InsulinCommand
    safety: SafetyDecision
    reward: float | None
    events: list[dict[str, Any]]
```

These records should become the common language between simulator, controllers, environments, evaluation, and UI.

## Simulator Core Responsibilities

The simulator core should:

- Own patient state and compartment integration.
- Apply meals, exercise, sensor, pump, and insulin delivery models.
- Advance simulation deterministically from a seed and config.
- Return structured `StepRecord` objects.
- Avoid controller-specific reward or RL logic.
- Avoid UI-specific state.
- Avoid filesystem writes during step/reset.

It should not:

- Know about A2C, PPO, TensorFlow, PyTorch, Streamlit, or Plotly.
- Store PID gain history unless a controller reports it in metadata.
- Select best models.
- Compute training rewards except through a supplied reward function.

## Controller Layer Responsibilities

Controllers should implement a stable interface:

```python
class Controller(Protocol):
    def reset(self, patient_config: PatientConfig, scenario: Scenario) -> None: ...
    def propose(self, observation: ControllerObservation) -> InsulinCommand: ...
```

Required controllers:

- `PIDController`: deterministic baseline with safe gains.
- `MPCController`: primary baseline and recommended core controller.
- `RLSupervisor`: slow learned policy that adjusts MPC parameters.
- `HybridController`: combines RL supervisor, MPC, and safety.
- `OracleController`: optional debug upper bound with perfect meal knowledge.
- `BasalOnlyController`: fallback and sanity baseline.

Controllers should return requested commands. The simulator should pass those through `SafetySupervisor` before delivery.

## RL/Training Layer Responsibilities

The RL layer should:

- Use Gymnasium-compatible environments.
- Own reward functions and observation/action transforms.
- Log config, seed, git commit, package versions, and metrics.
- Save checkpoints based on clinical evaluation criteria, not raw training reward.
- Support deterministic evaluation.

It should not:

- Depend on Streamlit.
- Mutate simulator internals.
- Use hidden global RNG state.
- Monkey-patch training functions.

## Safety Supervisor Responsibilities

The safety supervisor is the final authority before insulin delivery.

It should:

- Clamp basal and bolus.
- Enforce IOB and daily insulin limits.
- Suspend insulin below low glucose thresholds.
- Predict near-term hypoglycemia.
- Apply rate-of-change limits.
- Enforce bolus lockout.
- Return structured reasons for every modification.

It should not:

- Be trainable.
- Depend on reward function.
- Be bypassed by any controller.
- Silently modify commands without logging.

## Patient Modeling Responsibilities

Patient models should:

- Implement a common `PatientModel` interface.
- Expose full compartments for diagnostics.
- Support deterministic reset from config.
- Support patient parameter sampling for population studies.
- Keep ODE equations separate from integration.

Initial model:

- Hovorka 10-state model, refactored from current code.

Optional later models:

- Bergman minimal model.
- UVA/Padova-inspired virtual population if licensing and implementation are appropriate.
- Learned forecaster model for MPC prediction.

## Scenario Engine Responsibilities

The scenario engine should:

- Load structured YAML scenarios.
- Preserve compatibility with existing `data/test_scenarios`.
- Represent meals, exercise, stress, illness, sensor faults, pump faults, and carb-counting error.
- Support deterministic sampling with explicit seeds.
- Produce event streams in memory.

It should not:

- Write temp files.
- Depend on patient model internals.
- Use exact floating-point equality for event matching.

## Experiment Tracking Responsibilities

Every experiment should persist:

- Full config.
- Random seeds.
- Controller name and version.
- Patient profile or sampled parameters.
- Scenario ID or sampled scenario parameters.
- Git commit if available.
- Package versions.
- Training curves.
- Evaluation metrics.
- Safety events.
- Checkpoint path.
- Artifacts such as plots and replay files.

Use local JSON/CSV logging first. Add MLflow or Weights & Biases after the local format is stable.

## UI/Backend Responsibilities

The backend should:

- Expose simulation and evaluation APIs.
- Own long-running simulation execution.
- Stream step records for playback.
- Return typed responses.

The UI should:

- Edit scenarios.
- Launch simulations.
- Play, pause, step, and scrub trajectories.
- Display synchronized glucose, insulin, IOB, meals, exercise, and safety overlays.
- Compare controllers.
- Inspect controller decisions.

The UI should not instantiate low-level environment classes directly or monkey-patch their internals.

---

# 3. Implementation Phases

Do not build everything at once. Each phase below has one primary risk it removes. Later phases depend on earlier phases.

## Phase 0 - Repository Cleanup and Baseline Capture

### Objective

Create a clean baseline before changing behavior. The purpose is to know what exists, what is broken, and what the new system must preserve.

### Refactor or Rewrite

Refactor only. Do not change simulator behavior yet.

### Dependencies

None.

### Exact Tasks

1. Create an implementation branch.
2. Record current `git status --short`.
3. Run existing tests.
4. Run one baseline rollout using the current baseline PID.
5. Save a baseline artifact with glucose, insulin, PID gains, meals, exercise, reward, and stats.
6. Add a short `docs/current-system-notes.md` only if needed for human notes.
7. Do not delete legacy code yet. Mark delete candidates in the migration plan first.

### File-Level Actions

- Read but do not modify `src/ap_rl/envs/diabetes_pid_env.py`.
- Read but do not modify `src/ap_rl/envs/hovorka_patient.py`.
- Read but do not modify `src/ap_rl/utils/pid_controller.py`.
- Add no behavior changes in this phase unless a test cannot run due to an environment issue.

### Expected Deliverables

- Current test result.
- Current smoke rollout result.
- A known current failure list.
- A baseline trajectory file if a script already supports it; otherwise document command output.

### Testing Procedures

Run:

```bash
pytest
python scripts/smoke_baseline_rollout.py
```

If TensorFlow or optional Gym dependencies fail, isolate those failures and keep simulator tests separate.

### Acceptance Criteria

- Existing test suite status is known.
- At least one baseline rollout can be reproduced with a fixed seed, or the exact blocker is documented.
- No source behavior changed.

### Common Failure Modes

- TensorFlow import slows or breaks tests.
- Numba cache differences cause different trajectories.
- Checkpoint files are missing.
- Current tests may already encode outdated expectations, especially correction dose behavior.

### Rollback Considerations

No rollback should be needed because this phase should not mutate source behavior.

### Estimated Complexity

Low. One half day.

---

## Phase 1 - Core Simulation Stabilization

### Objective

Make the Hovorka simulator deterministic, numerically consistent, and testable. This is the first real implementation phase because all controllers and RL results depend on simulator correctness.

### Refactor or Rewrite

Refactor `src/ap_rl/envs/hovorka_patient.py` into a new simulation module while preserving equations. Do not rewrite the physiology from scratch.

### Dependencies

Phase 0 baseline capture.

### Exact Tasks

1. Create `src/ap_rl/simulation/`.
2. Add `src/ap_rl/simulation/hovorka.py`.
3. Move Hovorka ODE right-hand-side logic into a pure function:
   - Inputs: state vector, time, insulin input, exercise sensitivity, parameter object.
   - Output: derivative vector.
4. Add `src/ap_rl/simulation/integrators.py`.
5. Implement a fixed-step RK4 integrator.
6. Use the same integrator for all runtime paths.
7. Remove behavioral dependence on whether Numba is installed.
8. Include circadian EGP modulation in all paths.
9. Remove hidden default differences such as `t_max_G=40` in Python ODE vs `t_max_G=30` elsewhere.
10. Require all model parameters to come from a validated config object.
11. Add in-memory meal and exercise APIs.
12. Replace exact floating-point event matching with integer minute indices or tolerance.
13. Add non-negativity guards for state compartments after each internal sub-step.
14. Expose compartment state for diagnostics and IOB calculation.
15. Keep the old class as a compatibility wrapper only during migration, if necessary.

### File-Level Actions

- Add:
  - `src/ap_rl/simulation/__init__.py`
  - `src/ap_rl/simulation/hovorka.py`
  - `src/ap_rl/simulation/integrators.py`
  - `src/ap_rl/simulation/patient.py`
- Refactor from:
  - `src/ap_rl/envs/hovorka_patient.py`
- Keep temporarily:
  - `src/ap_rl/envs/hovorka_patient.py` as a compatibility import or wrapper.

### Expected Deliverables

- One Hovorka implementation with one integration path.
- Deterministic one-minute stepping from a fixed initial state.
- Reference comparison against SciPy RK45.
- No temp-file dependency for meals or exercise.

### Validation Strategy

Compare RK4 output against a high-accuracy RK45 reference for:

- Fasting basal-only day.
- Single 45 g meal without insulin.
- Single bolus without meal.
- Meal plus bolus.
- Exercise event.
- Dawn/circadian EGP interval.

Use tolerances:

- Glucose trajectory RMSE vs RK45 under 2 mg/dL for nominal scenarios.
- Maximum absolute glucose difference under 5 mg/dL for 24-hour nominal trajectories.
- No negative compartments after clamping.
- No NaN or infinity.

### Testing Strategy

Add:

- `tests/unit/test_hovorka_integrator.py`
- `tests/unit/test_hovorka_events.py`
- `tests/regression/test_golden_trajectories.py`

Test cases:

- `test_numba_presence_does_not_change_physics`: if Numba remains optional, importing with or without Numba must not change results.
- `test_circadian_egp_applied`: EGP changes by time of day in the same way for all paths.
- `test_tmax_g_config_required`: no silent fallback to inconsistent default.
- `test_meal_event_integer_index`: meal at minute 480 triggers exactly once.
- `test_no_temp_files_created_on_reset`.

### Acceptance Criteria

- The simulator has one authoritative integration path.
- Hovorka stepping no longer depends on `NUMBA_AVAILABLE`.
- Circadian EGP is present in every path.
- Existing env can still run through a compatibility wrapper or a temporary adapter.
- Golden trajectories are committed and stable.

### Common Failure Modes

- Accidentally changing units: U/h vs U/min, mmol/L vs mg/dL, grams vs mmol.
- Forgetting body-weight scaling for `V_G`, `V_I`, `F_01`, and `EGP_0`.
- Applying meal carbs on every sub-step instead of once at event time.
- Clamping compartments too aggressively and hiding numerical bugs.
- Comparing 1-minute output against RK45 with different event timing.

### Rollback Considerations

Keep the old `HovorkaPatient` implementation available behind a feature flag for one phase. If new golden tests fail badly, switch the compatibility wrapper back while fixing the new module.

### Estimated Complexity

High. Three to five days.

---

## Phase 2 - Safety Supervisor and Insulin Accounting

### Objective

Build a controller-independent safety layer that constrains total insulin delivery, not just basal insulin. Replace the exponential IOB estimate with a physiologically meaningful estimate.

### Refactor or Rewrite

Rewrite safety as a new module. Preserve useful bolus math from `InsulinCalculator`.

### Dependencies

Phase 1 compartment access and deterministic simulation.

### Exact Tasks

1. Add `src/ap_rl/controllers/safety.py`.
2. Define `SafetySupervisor`.
3. Define configuration fields:
   - `max_basal_u_h`
   - `max_bolus_u`
   - `max_iob_u`
   - `max_daily_insulin_u`
   - `min_bolus_interval_min`
   - `max_rate_change_u_h_per_min`
   - `suspend_glucose_mgdl`
   - `reduce_glucose_mgdl`
   - `predictive_horizon_min`
   - `predictive_suspend_threshold_mgdl`
4. Define a command-level API:
   - input: requested `InsulinCommand`, current patient state, recent glucose history, current IOB, recent bolus history.
   - output: delivered `InsulinCommand` plus `SafetyDecision`.
5. Compute IOB from Hovorka insulin compartments where possible.
6. If compartment-derived IOB is not immediately reliable, implement a validated two-compartment rapid-acting insulin model and compare both during migration.
7. Apply safety to basal and bolus together.
8. Add predictive low-glucose suspend:
   - simple linear projection first;
   - later replace or augment with MPC/forecaster prediction.
9. Add emergency logic:
   - severe hypo: all insulin off and safety event logged.
   - low and falling: all insulin off.
   - near-low and falling: reduce basal.
   - excessive IOB: block bolus and reduce/suspend basal.
10. Log every active constraint.

### File-Level Actions

- Add:
  - `src/ap_rl/controllers/safety.py`
  - `src/ap_rl/core/types.py`
  - `src/ap_rl/core/constants.py`
- Modify later:
  - `src/ap_rl/envs/diabetes_pid_env.py` only through adapter integration or replacement.
- Preserve:
  - `src/ap_rl/utils/insulin_calculator.py`, but do not allow it to bypass safety.

### Expected Deliverables

- A standalone safety supervisor with unit tests.
- Safety decisions visible in step records.
- No controller path can deliver insulin without safety pass-through.
- IOB estimate available in every step record.

### Validation Strategy

Run synthetic safety tests without full RL:

- Current glucose 50 mg/dL, requested bolus 10 U: delivered bolus must be 0.
- Current glucose 68 mg/dL, requested basal 2 U/h: delivered basal must be 0.
- Current glucose 85 mg/dL and falling at -2 mg/dL/min: suspend or strong reduction.
- IOB above cap: block new bolus.
- Requested basal above max: clamp to max.
- Repeated bolus within lockout: reject second bolus.
- Daily insulin above cap: block non-emergency insulin.

### Testing Strategy

Add:

- `tests/unit/test_safety_supervisor.py`
- `tests/integration/test_safety_closed_loop.py`

Property tests:

- Delivered basal is always `0 <= basal <= max_basal`.
- Delivered bolus is always `0 <= bolus <= max_bolus`.
- Delivered insulin is zero below suspend threshold.
- Safety never increases total insulin above requested insulin.

### Acceptance Criteria

- Basal-only clamp is gone from primary architecture.
- Total command safety is mandatory.
- Safety tests pass independent of RL and PID.
- Safety events are logged and inspectable.

### Common Failure Modes

- Treating bolus U as U/h.
- Clamping basal but forgetting queued tail bolus.
- Failing to include pending tail dose in IOB.
- Forgetting that pump delivery quantization can alter delivered dose.
- Making safety too aggressive and hiding controller problems during evaluation.

### Rollback Considerations

Keep current `_safety_clamp` only as a legacy compatibility behavior for old environment tests. New controllers must use `SafetySupervisor`.

### Estimated Complexity

Medium-high. Two to four days.

---

## Phase 3 - Controller Abstraction and Safe PID Baseline

### Objective

Separate controllers from environments and create a reliable baseline before adding MPC or RL.

### Refactor or Rewrite

Rewrite controller interfaces. Replace or heavily refactor the existing PID.

### Dependencies

Phases 1 and 2.

### Exact Tasks

1. Add `src/ap_rl/controllers/base.py`.
2. Define a controller protocol with `reset` and `propose`.
3. Add `src/ap_rl/controllers/pid.py`.
4. Implement discrete-time PID with explicit `dt_min`.
5. Remove `time.time()` from control logic.
6. Initialize `PTerm = 0.0`, not `0.2`.
7. Add derivative low-pass filtering.
8. Add asymmetric anti-windup.
9. Add output limits in insulin units.
10. Use controller metadata for PID terms and gains.
11. Add `BasalOnlyController`.
12. Add `OracleMealController` only for debugging and upper-bound comparisons.

### File-Level Actions

- Add:
  - `src/ap_rl/controllers/__init__.py`
  - `src/ap_rl/controllers/base.py`
  - `src/ap_rl/controllers/pid.py`
  - `src/ap_rl/controllers/oracle.py`
- Deprecate:
  - `src/ap_rl/utils/pid_controller.py`
- Modify tests:
  - replace direct assumptions about old PID internals.

### Expected Deliverables

- Safe deterministic PID baseline.
- Basal-only fallback.
- Controller interface used by simulator runner.
- No wall-clock time dependency.

### Validation Strategy

Validate PID in isolation:

- Zero error produces zero adjustment after reset.
- Positive hyperglycemia error increases insulin in the correct direction.
- Hypoglycemia error reduces insulin or requests zero.
- Integral term does not wind up during saturation.
- Derivative term is bounded under sensor noise.

Validate closed-loop:

- Basal-only does not crash nominal fasting scenario.
- PID baseline can handle a standard meal day without severe hypo under safety supervisor.
- Safety supervisor still blocks unsafe PID outputs.

### Testing Strategy

Add:

- `tests/unit/test_pid_controller.py`
- `tests/integration/test_pid_baseline_closed_loop.py`

### Acceptance Criteria

- Old `PID` is not used by new controllers.
- No controller code calls wall-clock time.
- PID output is expressed in insulin command units, not arbitrary PID output scaled by magic `0.01`.
- PID metadata can be plotted without environment-specific fields.

### Common Failure Modes

- Sign inversion: glucose above target must increase insulin, glucose below target must reduce insulin.
- Double-scaling output by both PID limits and basal conversion.
- Integral accumulation while safety is clamping output.
- Derivative kick on setpoint changes.

### Rollback Considerations

Keep legacy PID-delta environment runnable until new PID baseline passes closed-loop tests. Do not use it for new research.

### Estimated Complexity

Medium. Two to three days.

---

## Phase 4 - Simulation Runner and Step Records

### Objective

Create a framework-neutral runner that connects scenario, patient, controller, safety, sensor, pump, metrics, and logs. This runner becomes the backend for evaluation, RL environments, and UI.

### Refactor or Rewrite

Rewrite orchestration. Preserve `EpisodeRecord` ideas from `runtime/rollout.py`.

### Dependencies

Phases 1 to 3.

### Exact Tasks

1. Add `src/ap_rl/simulation/simulator.py`.
2. Add `SimulationConfig`.
3. Add `SimulationRunner`.
4. Use explicit seeded RNG objects.
5. Return a list or iterator of `StepRecord`.
6. Add `EpisodeRecordV2` or generalize the existing `EpisodeRecord`.
7. Include:
   - true glucose;
   - observed glucose;
   - requested insulin;
   - delivered insulin;
   - basal/bolus decomposition;
   - IOB;
   - COB;
   - safety events;
   - controller metadata;
   - scenario events.
8. Make `runtime/rollout.py` a thin compatibility layer around the new runner.
9. Remove top-level TensorFlow actor import from rollout path.

### File-Level Actions

- Add:
  - `src/ap_rl/simulation/simulator.py`
  - `src/ap_rl/runtime/records.py`
- Modify:
  - `src/ap_rl/runtime/rollout.py`
- Avoid modifying:
  - UI until runner is stable.

### Expected Deliverables

- One deterministic runner for non-RL simulations.
- Structured records for plotting and evaluation.
- Compatibility path for old demo if needed.

### Validation Strategy

Run the same seeded simulation twice and compare records exactly or within floating tolerance.

### Testing Strategy

Add:

- `tests/integration/test_simulation_runner.py`
- `tests/integration/test_reproducibility.py`

### Acceptance Criteria

- Same config and seed produce identical trajectories.
- No file writes during reset or step.
- TensorFlow is not imported for baseline simulation.
- Step records contain enough information to rebuild UI charts without accessing environment internals.

### Common Failure Modes

- Hidden use of global `np.random`.
- Controller metadata not serializable.
- Step record growing too large because full compartments are stored every sub-step instead of every output step.

### Rollback Considerations

Keep old `run_episode` API as wrapper until UI and tests move over.

### Estimated Complexity

Medium. Two to four days.

---

## Phase 5 - Gymnasium Environment Redesign

### Objective

Replace the current PID-delta environment with standard Gymnasium environments that expose meaningful action spaces and stable observations.

### Refactor or Rewrite

Rewrite. The current `DiabetesPIDEnv` should not remain the primary RL environment.

### Dependencies

Phases 1 to 4.

### Exact Tasks

1. Add `src/ap_rl/environments/`.
2. Delete/rebuild `src/ap_rl/envs/hovorka_gym_env.py`.
3. Implement `GlucoseControlEnv`:
   - action: direct basal adjustment or insulin command within safe bounds;
   - safety supervisor still clamps delivered command.
4. Implement `SupervisoryEnv`:
   - action every 15-30 minutes;
   - action adjusts MPC target, aggressiveness, insulin cost, sensitivity multiplier, or meal bolus fraction.
5. Use Gymnasium API:
   - `reset(seed=None, options=None) -> (obs, info)`;
   - `step(action) -> (obs, reward, terminated, truncated, info)`.
6. Define observations from physiology, not PID internals:
   - recent glucose history;
   - glucose rate;
   - IOB;
   - COB;
   - time of day sin/cos;
   - meal announcement or detected meal flag;
   - exercise status;
   - patient sensitivity features;
   - previous delivered insulin.
7. Add normalization wrappers.
8. Remove redundant near-zero features.
9. Ensure observation spaces and action spaces are single sources of truth.
10. Add deterministic seeding for patient and scenario sampling.

### File-Level Actions

- Add:
  - `src/ap_rl/environments/__init__.py`
  - `src/ap_rl/environments/glucose_control_env.py`
  - `src/ap_rl/environments/supervisory_env.py`
  - `src/ap_rl/environments/wrappers.py`
- Deprecate:
  - `src/ap_rl/envs/diabetes_pid_env.py`
  - `src/ap_rl/envs/hovorka_gym_env.py`

### Expected Deliverables

- Gymnasium-compatible direct insulin environment.
- Gymnasium-compatible supervisory environment.
- Clear observation and action definitions.
- Old PID-gain-delta action space removed from primary training.

### Validation Strategy

Run Gymnasium environment checker.

Check:

- Observation matches declared shape and dtype.
- Action clipping is deterministic.
- `terminated` is reserved for physiological terminal events if used.
- `truncated` handles horizon limits.
- `info` contains clinical metrics and safety events.

### Testing Strategy

Add:

- `tests/unit/test_env_spaces.py`
- `tests/integration/test_gymnasium_env.py`
- `tests/integration/test_env_seed_reproducibility.py`

### Acceptance Criteria

- Training code can instantiate envs without legacy `DiabetesPIDEnv`.
- `hovorka_gym_env.py` is gone or a compatibility alias only.
- No observation includes hidden controller state unless the environment is explicitly a controller-tuning environment.
- Safety supervisor always applies after agent action.

### Common Failure Modes

- Letting RL action represent requested insulin but computing reward from unclamped insulin.
- Normalizing observations inconsistently between training and evaluation.
- Hiding current IOB from the agent while penalizing IOB-related outcomes.
- Returning clean true glucose in reward but noisy observations without documenting the POMDP.

### Rollback Considerations

Keep legacy environment available under `legacy/` for checkpoint replay only. Do not train new policies against it.

### Estimated Complexity

High. Four to seven days.

---

## Phase 6 - Clinical Reward, Observation, and Action Specification

### Objective

Replace chaotic reward shaping with bounded clinical objectives and document observation/action semantics.

### Refactor or Rewrite

Rewrite reward. Rewrite observations and actions for new environments.

### Dependencies

Phase 5.

### Exact Tasks

1. Add `src/ap_rl/rewards/clinical.py`.
2. Implement a bounded reward with:
   - positive reward for 70-180 mg/dL;
   - stronger penalty for 54-70 mg/dL;
   - severe but bounded penalty for below 54 mg/dL;
   - smaller penalty for 180-250 mg/dL;
   - moderate penalty above 250 mg/dL;
   - small smoothness penalty for insulin changes.
3. Keep reward roughly within `[-30, +1]` per step.
4. Remove catastrophic `-10000` reward spikes.
5. Remove perverse recovery bonus that rewards allowing spikes then recovering.
6. Remove conflict between reward target 100 mg/dL and controller target 120 mg/dL.
7. Add optional multi-objective reward vector for research later.
8. Document each observation feature:
   - name;
   - units;
   - normalization;
   - expected range;
   - whether noisy or true.
9. Document each action:
   - physical meaning;
   - units;
   - bounds;
   - clipping behavior;
   - safety interaction.

### File-Level Actions

- Add:
  - `src/ap_rl/rewards/__init__.py`
  - `src/ap_rl/rewards/clinical.py`
  - `docs/rl_environment.md`
- Modify:
  - new environments to call reward module.

### Expected Deliverables

- Reward function aligned with clinical outcomes.
- Reward tests.
- Observation/action spec document.

### Validation Strategy

Plot reward vs glucose and inspect:

- Reward is flat or gently peaked in safe range.
- Hypo penalty dominates hyper penalty.
- Severe hypo does not create numerical instability.
- Reward changes smoothly at thresholds or threshold discontinuities are intentional and tested.

### Testing Strategy

Add:

- `tests/unit/test_clinical_reward.py`
- `tests/unit/test_observation_builder.py`
- `tests/unit/test_action_mapping.py`

Test reward monotonicity:

- 50 mg/dL is worse than 60 mg/dL.
- 60 mg/dL is worse than 80 mg/dL.
- 300 mg/dL is worse than 200 mg/dL.
- 55 mg/dL penalty is more severe than 250 mg/dL penalty.

### Acceptance Criteria

- Reward is bounded.
- Reward is not anti-correlated with TIR on smoke scenarios.
- Observation normalization is documented and tested.
- Action mapping is physical and auditable.

### Common Failure Modes

- Reintroducing complex hand-tuned shaping that dominates TIR.
- Penalizing insulin so heavily that the agent learns to underdose.
- Rewarding exact target glucose instead of safe range.
- Using different reward during training and evaluation without logging it.

### Rollback Considerations

Keep old reward only for reproducing old checkpoints. New checkpoints must use reward v2.

### Estimated Complexity

Medium. Two to three days.

---

## Phase 7 - Metrics Engine and Evaluation Harness

### Objective

Make evaluation clinically meaningful, deterministic, and independent of training reward.

### Refactor or Rewrite

Refactor `src/ap_rl/evaluation/metrics.py`; add new evaluation modules.

### Dependencies

Phases 1 to 6.

### Exact Tasks

1. Expand clinical metrics:
   - TIR 70-180;
   - tight TIR 80-140 as secondary only;
   - TBR 54-70;
   - TBR below 54;
   - TAR 180-250;
   - TAR above 250;
   - mean glucose;
   - median glucose;
   - standard deviation;
   - coefficient of variation;
   - GMI;
   - LBGI;
   - HBGI;
   - MAGE if feasible;
   - area under hypo/hyper excursions;
   - max glucose rate of change;
   - time to return to range after meals.
2. Add safety metrics:
   - peak IOB;
   - total daily insulin;
   - max basal;
   - max bolus;
   - number of safety suspensions;
   - duration of suspension;
   - predicted low events;
   - actual low events;
   - false positive/false negative predictive suspend events.
3. Add `EvaluationHarness`.
4. Add fixed scenario suites:
   - smoke;
   - nominal;
   - held-out;
   - stress;
   - adversarial.
5. Add Monte Carlo evaluation over patients and scenarios.
6. Add paired controller comparison.
7. Add pass/fail thresholds.

### File-Level Actions

- Modify:
  - `src/ap_rl/evaluation/metrics.py`
- Add:
  - `src/ap_rl/evaluation/harness.py`
  - `src/ap_rl/evaluation/monte_carlo.py`
  - `src/ap_rl/evaluation/stress_tests.py`
  - `src/ap_rl/evaluation/comparison.py`
  - `src/ap_rl/evaluation/golden.py`

### Expected Deliverables

- Clinical metrics dictionary for every run.
- Evaluation CLI.
- Controller comparison reports.
- Regression suite for golden trajectories.

### Validation Strategy

Metric sanity checks:

- Constant 100 mg/dL trace: TIR 100%, TBR 0%, TAR 0%, CV 0%.
- Constant 50 mg/dL trace: TBR below 54 is 100%.
- Mixed synthetic trace: percentages sum correctly.
- GMI formula uses mg/dL.
- LBGI/HBGI risk increases in correct direction.

### Testing Strategy

Add:

- `tests/unit/test_clinical_metrics.py`
- `tests/integration/test_evaluation_harness.py`
- `tests/regression/test_golden_reports.py`

### Acceptance Criteria

- Best checkpoint selection can use TIR/TBR/CV and not mean reward.
- Evaluation report includes clinical metrics and safety metrics.
- Same controller and seed produce identical evaluation report.

### Common Failure Modes

- Reporting TIR 80-140 as if it were clinical consensus TIR.
- Ignoring severe hypo because average TIR is good.
- Mixing sample frequency assumptions.
- Comparing controllers on different sampled scenarios.

### Rollback Considerations

Keep old `glucose_trajectory_summary` keys for compatibility, but mark them legacy.

### Estimated Complexity

Medium. Three to five days.

---

## Phase 8 - Training Infrastructure and Reproducibility

### Objective

Replace ad-hoc A2C training and monkey-patching with reproducible trainer infrastructure.

### Refactor or Rewrite

Rewrite training orchestration. Migrate away from raw TensorFlow A2C unless there is a strong reason to keep it for legacy checkpoints.

### Dependencies

Phases 5 to 7.

### Exact Tasks

1. Add `src/ap_rl/training/trainer.py`.
2. Add experiment config schema.
3. Add deterministic seeding:
   - Python random;
   - NumPy `Generator`;
   - PyTorch if migrated;
   - environment seed;
   - scenario seed;
   - patient seed.
4. Replace global `np.random.normal()` calls with injected RNG.
5. Remove monkey-patched `agent.train` logic from `train_a2c.py`.
6. Add callbacks:
   - checkpoint;
   - evaluation;
   - early stopping;
   - metrics logging;
   - safety event logging.
7. Use Stable-Baselines3 PPO/SAC or CleanRL-style PyTorch implementation.
8. Save checkpoints by clinical evaluation score:
   - primary: TBR < threshold;
   - then maximize TIR;
   - then minimize CV and total insulin.
9. Dump full config with every run.
10. Add local experiment directory format.

### File-Level Actions

- Add:
  - `src/ap_rl/training/trainer.py`
  - `src/ap_rl/training/experiment.py`
  - `src/ap_rl/training/callbacks.py`
  - `configs/training/ppo_default.yaml`
  - `configs/training/sac_default.yaml`
- Refactor:
  - `src/ap_rl/training/train_a2c.py`
  - `src/ap_rl/agents/diabetes_a2c_agent.py`
- Possibly delete later:
  - TensorFlow-specific `src/ap_rl/utils/hardware.py`
  - `src/ap_rl/utils/checkpoint_filenames.py` if Keras checkpoints are no longer supported.

### Expected Deliverables

- Reproducible training command.
- Structured run directories.
- No monkey-patching.
- No global RNG use in training.
- Checkpoint chosen by clinical metrics.

### Suggested Run Directory

```text
runs/
  2026-05-15_120000_ppo_supervisory/
    config.yaml
    git.json
    environment.json
    metrics.csv
    safety_events.csv
    checkpoints/
      latest.pt
      best_clinical.pt
    eval/
      heldout_metrics.json
      stress_metrics.json
      plots/
```

### Validation Strategy

Run a two-episode training smoke test twice with the same seed. Initial actions, rewards, and metrics should match.

### Testing Strategy

Add:

- `tests/integration/test_training_smoke.py`
- `tests/integration/test_training_reproducibility.py`
- `tests/unit/test_callbacks.py`

### Acceptance Criteria

- Training can run without modifying source files.
- Two seeded smoke runs match.
- Best checkpoint selection does not use raw rolling reward alone.
- Evaluation can run without training code imported.

### Common Failure Modes

- Deep learning backend nondeterminism.
- Accidentally using vectorized envs with shared temp paths.
- Logging only stdout and losing config.
- Evaluating on training scenarios.
- Selecting high TIR policies that violate TBR severe hypo thresholds.

### Rollback Considerations

Keep old A2C only as `legacy_a2c` for checkpoint replay. Do not keep it as the default training path.

### Estimated Complexity

High. One to two weeks depending on framework migration.

---

## Phase 9 - MPC Primary Controller

### Objective

Implement a strong non-RL controller that handles delayed insulin dynamics and constraints. This becomes the primary control baseline and the expert for imitation learning.

### Refactor or Rewrite

New implementation.

### Dependencies

Phases 1 to 8, especially safety and evaluation.

### Exact Tasks

1. Add `src/ap_rl/controllers/mpc.py`.
2. Start with a simplified MPC:
   - prediction horizon: 180 minutes;
   - control horizon: 30 minutes;
   - control interval: 5 minutes;
   - decision variable: basal profile or basal adjustment;
   - optional correction bolus decision later.
3. Use Hovorka or a linearized/approximated model for prediction.
4. Define cost:
   - glucose target deviation;
   - asymmetric hypo penalty;
   - insulin smoothness;
   - IOB penalty;
   - terminal risk penalty.
5. Add hard constraints:
   - basal bounds;
   - bolus bounds;
   - IOB cap;
   - rate-of-change constraints.
6. Use `scipy.optimize.minimize` first if simpler.
7. Consider `cvxpy` for QP formulation if model is linearized.
8. Cache previous solution for warm start.
9. Return interpretable metadata:
   - predicted glucose trajectory;
   - planned insulin trajectory;
   - active constraints;
   - solver status;
   - objective components.
10. Ensure safety supervisor still runs after MPC.

### File-Level Actions

- Add:
  - `src/ap_rl/controllers/mpc.py`
  - `configs/controllers/mpc_default.yaml`
- Modify:
  - evaluation harness to include MPC.

### Expected Deliverables

- MPC controller runnable on nominal scenarios.
- MPC metadata available for UI and debug.
- MPC baseline compared against PID.

### Validation Strategy

Evaluate MPC vs safe PID:

- Nominal day.
- Heavy meal day.
- Exercise day.
- Overnight scenario.
- Insulin-sensitive profile.
- Insulin-resistant profile.

Expected behavior:

- Lower severe hypo than aggressive PID.
- Better post-meal recovery than basal-only.
- Smooth insulin profile.
- Solver failures handled by fallback.

### Testing Strategy

Add:

- `tests/unit/test_mpc_constraints.py`
- `tests/integration/test_mpc_closed_loop.py`
- `tests/integration/test_mpc_fallback.py`

### Acceptance Criteria

- MPC solves within acceptable runtime for a 24-hour simulation.
- MPC never bypasses safety.
- Solver failure triggers fallback controller.
- MPC beats basal-only and is competitive with PID on TIR while improving safety.

### Common Failure Modes

- MPC too slow for UI or training.
- Prediction model mismatch causes over-delivery.
- Optimizer finds numerically valid but clinically aggressive commands.
- Cost weights accidentally reward underdosing to avoid hypo.

### Rollback Considerations

Fallback chain must be:

```text
MPC -> safe PID -> basal-only -> suspend
```

If MPC fails, closed-loop simulation must continue safely.

### Estimated Complexity

High. One to two weeks.

---

## Phase 10 - Hybrid MPC+RL Supervisor

### Objective

Reintroduce RL only where it is useful: slow adaptation of a constrained MPC controller.

### Refactor or Rewrite

New RL formulation. Do not adapt PID gains.

### Dependencies

Phases 8 and 9.

### Exact Tasks

1. Add `src/ap_rl/controllers/rl_supervisor.py`.
2. Add `src/ap_rl/controllers/hybrid.py`.
3. Use `SupervisoryEnv`.
4. RL action every 15-30 minutes.
5. Action controls bounded supervisory variables:
   - target glucose adjustment, e.g. 100-140 mg/dL;
   - aggressiveness scalar;
   - insulin cost multiplier;
   - sensitivity multiplier;
   - meal bolus fraction.
6. Observation includes:
   - recent 2-hour glucose history;
   - IOB;
   - COB;
   - time of day;
   - meal announcement;
   - exercise status;
   - current MPC settings;
   - patient sensitivity estimate.
7. Train with PPO or SAC.
8. Pretrain supervisor from MPC/expert behavior if helpful.
9. Evaluate against MPC-only, PID, and basal-only.
10. Keep safety supervisor mandatory.

### File-Level Actions

- Add:
  - `src/ap_rl/controllers/rl_supervisor.py`
  - `src/ap_rl/controllers/hybrid.py`
  - `configs/controllers/hybrid_mpc_rl.yaml`
  - `configs/training/supervisory_ppo.yaml`
- Modify:
  - `src/ap_rl/environments/supervisory_env.py`
  - training configs.

### Expected Deliverables

- Hybrid controller.
- Supervisory RL policy checkpoint.
- Evaluation report against MPC-only.

### Validation Strategy

RL should improve or match MPC without increasing safety violations.

Minimum acceptance:

- TBR <54 does not increase versus MPC.
- TIR improves or remains within confidence interval.
- Total insulin does not become clinically implausible.
- Safety intervention count does not explode.

### Testing Strategy

Add:

- `tests/integration/test_hybrid_controller_smoke.py`
- `tests/integration/test_supervisory_env.py`

### Acceptance Criteria

- No PID gain delta actions remain in primary RL path.
- Hybrid controller runs deterministically in evaluation mode.
- Hybrid fails safely when RL checkpoint is missing or invalid.
- Evaluation compares paired scenarios.

### Common Failure Modes

- RL learns to push MPC weights to extremes.
- Supervisor action bounds too wide.
- Training reward improves while TBR worsens.
- Safety layer hides unsafe learned behavior, making raw requested commands dangerous.

### Rollback Considerations

Hybrid must be optional. MPC-only remains the production-quality baseline. If RL supervisor fails validation, ship MPC and safety first.

### Estimated Complexity

High. Two to three weeks after MPC exists.

---

## Phase 11 - Scenario Generation and Population Modeling

### Objective

Improve generalization by expanding patient and scenario variability.

### Refactor or Rewrite

Refactor current scenario builder and add population sampling.

### Dependencies

Phases 1, 4, and 7.

### Exact Tasks

1. Add `src/ap_rl/simulation/scenario.py`.
2. Preserve `configs/meals/*.yaml`.
3. Add `configs/scenarios/`.
4. Create structured scenario schema:
   - meals;
   - exercise;
   - stress;
   - illness;
   - sensor dropout;
   - pump occlusion;
   - carb counting error;
   - absorption variability.
5. Add patient population sampling:
   - body weight;
   - insulin sensitivity;
   - EGP;
   - carb absorption;
   - insulin absorption;
   - circadian amplitude.
6. Use log-normal or bounded distributions.
7. Add deterministic sampled scenario IDs.
8. Add stress scenario catalog.

### File-Level Actions

- Add:
  - `src/ap_rl/simulation/scenario.py`
  - `src/ap_rl/simulation/population.py`
  - `configs/scenarios/normal_day.yaml`
  - `configs/scenarios/heavy_meals.yaml`
  - `configs/scenarios/exercise_day.yaml`
  - `configs/scenarios/overnight.yaml`
  - `configs/scenarios/stress_test.yaml`
- Preserve:
  - `src/ap_rl/envs/scenario_builder.py` as compatibility or move logic.

### Expected Deliverables

- Structured scenarios.
- Patient sampler.
- Stress scenario suite.
- Backward compatibility with `data/test_scenarios`.

### Validation Strategy

Scenario validation should reject:

- negative carbs;
- impossible times;
- overlapping invalid events;
- body weight outside plausible bounds;
- negative physiological parameters.

### Testing Strategy

Add:

- `tests/unit/test_scenario_schema.py`
- `tests/unit/test_patient_population.py`
- `tests/integration/test_stress_suite_runs.py`

### Acceptance Criteria

- Monte Carlo evaluation can sample patients and scenarios deterministically.
- Legacy data can still be loaded.
- Scenario events are inspectable in step records.

### Common Failure Modes

- Generating physiologically impossible patients.
- Making scenario distributions too easy.
- Training and evaluating on identical sampled seeds.

### Rollback Considerations

Keep fixed scenarios as canonical regression tests even after stochastic sampling is added.

### Estimated Complexity

Medium. Three to five days.

---

## Phase 12 - UI and Backend Rewrite

### Objective

Rebuild the demo around stable simulation APIs instead of directly instantiating environment internals.

### Refactor or Rewrite

Rewrite `app/app.py` after backend APIs are stable.

### Dependencies

Phases 4, 7, 9, and ideally 10.

### Exact Tasks

1. Add `src/ap_rl/app/server.py` with FastAPI.
2. Add `src/ap_rl/app/schemas.py` for request/response models.
3. Expose:
   - `POST /simulate`;
   - `POST /simulate/stream`;
   - `POST /evaluate`;
   - `GET /profiles`;
   - `GET /scenarios`;
   - `GET /controllers`.
4. Keep Streamlit as a thin client or replace with React later.
5. Remove direct imports of low-level env classes from UI.
6. Remove `env._skip_reload` monkey-patch.
7. Add playback state:
   - play;
   - pause;
   - step;
   - reset;
   - speed;
   - scrubber.
8. Add synchronized charts:
   - glucose;
   - insulin basal/bolus;
   - IOB/COB;
   - meals/exercise;
   - safety events;
   - controller predictions.
9. Add comparison mode:
   - PID vs MPC vs Hybrid;
   - same patient/scenario/seed.
10. Add debug overlays:
   - MPC prediction horizon;
   - active safety constraints;
   - controller requested vs delivered insulin;
   - sensor reading vs true glucose.
11. Add scenario editor with validation.
12. Add graceful error boundaries.

### File-Level Actions

- Add:
  - `src/ap_rl/app/server.py`
  - `src/ap_rl/app/schemas.py`
  - `src/ap_rl/app/streamlit_app.py`
  - `src/ap_rl/app/components/`
- Replace:
  - `app/app.py` with a compatibility launcher or remove after migration.

### Expected Deliverables

- UI does not know simulator internals.
- Live or streamed playback.
- Controller comparison.
- Debuggable safety and prediction overlays.

### Validation Strategy

Run a fixed simulation through backend and UI. Confirm the UI displays exactly the same metrics as evaluation harness.

### Testing Strategy

Add:

- API tests for simulation endpoints.
- Snapshot-like tests for response schema.
- Optional browser tests for major UI flows after server exists.

### Acceptance Criteria

- UI can run baseline PID, MPC, and hybrid without code changes.
- Missing checkpoint does not crash UI.
- Simulation failure returns a readable error.
- Charts remain synchronized during playback and scrubbing.

### Common Failure Modes

- Recreating monolith inside FastAPI endpoint.
- Long simulations blocking UI without progress.
- Streamlit reruns re-executing expensive simulations.
- Comparing controllers with different random seeds.

### Rollback Considerations

Keep old Streamlit file until new UI can run the main demo scenario.

### Estimated Complexity

High. One to two weeks.

---

# 4. Detailed Subsystem Plans

## Simulation Engine

### Architecture

The simulation engine should be the central deterministic runtime. It should own the patient model, scenario engine, sensor model, pump model, and safety-applied delivery loop.

Recommended files:

```text
src/ap_rl/simulation/
  simulator.py
  patient.py
  hovorka.py
  integrators.py
  insulin_pk.py
  meal_model.py
  exercise.py
  sensor.py
  pump.py
  scenario.py
```

### APIs and Interfaces

`SimulationRunner.run(config, controller) -> EpisodeRecord`.

`SimulationRunner.iter_steps(config, controller) -> Iterator[StepRecord]`.

The runner should accept a controller object, not a string that triggers framework imports.

### State Definitions

State must separate:

- true physiological state;
- observed sensor state;
- controller internal state;
- safety state;
- scenario event state.

Never hide controller gain state inside patient state.

### Timing Assumptions

Use internal sub-steps for ODE accuracy and output records at one-minute or five-minute intervals. The default should be one-minute simulation output to preserve compatibility, with metrics able to resample.

### Threading and Event-Loop Considerations

The simulator should be synchronous and deterministic by default. Parallelism should happen at the evaluation/training level by creating independent simulator instances with independent seeds. No shared temp paths or module-level mutable state.

### Numerical Stability Concerns

- Use RK4 fixed sub-steps initially.
- Guard against negative compartments.
- Fail fast on NaN or infinity.
- Validate against RK45.
- Keep units explicit.

### Configuration Strategy

Use typed config objects loaded from YAML. Do not use nested unvalidated dictionaries deep in the simulator.

### Logging Strategy

Every run logs:

- simulator version;
- integrator type;
- dt;
- patient parameters;
- scenario ID;
- seed;
- safety config.

### Testing Strategy

Golden trajectory tests are mandatory.

---

## Patient Model

### Architecture

Define `PatientModel` protocol and implement `HovorkaPatientModel`. Keep ODE equations pure and separate from integration.

### APIs

```python
class PatientModel:
    def reset(self) -> PatientState: ...
    def step(self, insulin_u_min: float, exogenous: ExogenousInputs, dt_min: float) -> PatientState: ...
    def get_compartments(self) -> dict[str, float]: ...
```

### State Definitions

Hovorka compartments:

- `S1`, `S2`: subcutaneous insulin compartments.
- `I`: plasma insulin.
- `x1`, `x2`, `x3`: insulin action compartments.
- `Q1`, `Q2`: glucose masses.
- `D1`, `D2`: gut absorption compartments.

Derived:

- glucose mg/dL;
- glucose mmol/L;
- IOB estimate;
- COB estimate.

### Timing Assumptions

Do not use exact equality on floating timestamps. Use integer minute event bins.

### Numerical Stability Concerns

The insulin dynamics are delayed and can be stiff enough that one-minute Euler is not acceptable. Use RK4 with sub-steps or a validated adaptive solver for reference.

### Configuration Strategy

All Hovorka parameters must be explicit after config validation. No ODE function should have hidden defaults that can diverge.

### Logging Strategy

Expose compartments in debug mode. Store only selected compartments in normal run logs to avoid huge artifacts.

### Testing Strategy

- Reference RK45 comparisons.
- Unit tests for circadian EGP.
- Unit tests for renal clearance threshold.
- Unit tests for meal absorption.
- Unit tests for exercise sensitivity.

---

## Meal and Exercise Modeling

### Architecture

Meals and exercise should be exogenous event models owned by the scenario engine, not file readers inside the patient.

### APIs

```python
class Scenario:
    def events_at(self, time_min: int) -> list[ScenarioEvent]: ...
    def exogenous_at(self, time_min: int) -> ExogenousInputs: ...
```

### State Definitions

Meal event:

- time;
- announced carbs;
- actual carbs;
- glycemic index;
- fat/protein modifier;
- absorption duration;
- announcement time if different from eating time.

Exercise event:

- start;
- duration;
- intensity;
- announced/unannounced flag;
- delayed sensitivity effect.

### Timing Assumptions

Meals and exercise should align to integer minutes. If a UI allows arbitrary time, convert at scenario validation.

### Numerical Stability Concerns

Avoid instantaneous unrealistic carb dumps as the only model. Start by preserving existing D1 injection, then add configurable absorption windows.

### Configuration Strategy

Use YAML scenario templates with deterministic randomization fields:

```yaml
meals:
  - time_min: 480
    carbs_g: 45
    absorption_min: 60
    announced: true
```

### Logging Strategy

Log actual and announced carbs separately. This matters for carb-counting error and controller observability.

### Testing Strategy

- Meal triggers once.
- Zero-carb meals are ignored.
- Exercise starts and ends correctly.
- Randomized scenarios reproduce with same seed.

---

## Insulin Kinetics

### Architecture

Insulin kinetics should be explicit. The patient model includes subcutaneous insulin compartments, but safety and UI need a clear IOB estimate.

### APIs

```python
class IOBModel:
    def update(delivered: InsulinCommand, patient_state: PatientState) -> float: ...
```

### State Definitions

Track:

- delivered basal insulin;
- delivered bolus insulin;
- pending tail bolus;
- active insulin estimate;
- insulin action forecast if available.

### Timing Assumptions

Rapid-acting insulin has delayed onset, peak, and long tail. A 45-minute exponential half-life is not enough.

### Numerical Stability Concerns

Do not subtract active insulin too quickly. Underestimated IOB causes insulin stacking.

### Configuration Strategy

Patient profiles should configure insulin absorption parameters, not hardcode a global decay.

### Logging Strategy

Log requested insulin, delivered insulin, IOB, and insulin action estimate separately.

### Testing Strategy

- Bolus IOB rises and decays over hours, not minutes.
- Basal IOB reaches steady state.
- IOB never goes negative.

---

## Safety Constraints

### Architecture

Safety is a mandatory command filter. It lives below all controllers.

### APIs

```python
SafetySupervisor.constrain(command, state, history) -> SafetyDecision
```

### State Definitions

Safety state includes:

- recent glucose history;
- current IOB;
- daily insulin total;
- last bolus time;
- active suspension status;
- predicted minimum glucose.

### Timing Assumptions

Safety runs every simulator output minute even if MPC or RL acts less frequently.

### Numerical Stability Concerns

Prediction should be conservative when data are missing or noisy. Missing CGM should trigger fallback behavior.

### Configuration Strategy

Use conservative defaults and profile-specific overrides only when justified.

### Logging Strategy

Every clamp or suspension must produce a named reason.

### Testing Strategy

Property tests for all insulin bounds and hypo behavior.

---

## RL Environment

### Architecture

Use Gymnasium as the primary RL interface. Provide separate environments for direct insulin control and supervisory control.

### APIs

`GlucoseControlEnv`: direct low-level action.

`SupervisoryEnv`: slow high-level action over MPC.

### State Definitions

Observation should be normalized and documented. Avoid PID internals unless training a PID-tuning environment for legacy comparison only.

### Timing Assumptions

Direct env acts every 5 minutes. Supervisory env acts every 15-30 minutes while MPC and safety continue internally.

### Threading Considerations

Vectorized training must create independent simulator instances. No shared files.

### Numerical Stability Concerns

Reward and observations must remain finite even during severe excursions.

### Configuration Strategy

Env config should include horizon, action mode, observation mode, reward version, sensor mode, patient sampler, and scenario sampler.

### Logging Strategy

The `info` dict should include clinical metrics so far, safety events, requested vs delivered insulin, and scenario events.

### Testing Strategy

Use Gymnasium checker and deterministic replay.

---

## Reward Function

### Architecture

Reward functions live in `src/ap_rl/rewards/`.

### APIs

```python
clinical_reward(step_record, previous_step_record, config) -> float
```

### State Definitions

Reward consumes step records. It should not mutate environment state.

### Timing Assumptions

Scale reward consistently by step duration if environments use different action intervals.

### Numerical Stability Concerns

Bound reward. Avoid extreme constants such as `-10000`.

### Configuration Strategy

Version rewards and save reward config with checkpoints.

### Logging Strategy

Log reward components separately during training.

### Testing Strategy

Plot and unit test reward curves.

---

## Observation Space

### Architecture

Create an observation builder for each environment.

### APIs

```python
ObservationBuilder.build(history, current_state, controller_context) -> np.ndarray
```

### State Definitions

Recommended supervisory observation:

- glucose history over last 2 hours;
- current glucose;
- glucose rate;
- IOB;
- COB;
- time sin/cos;
- exercise active;
- meal announcement flag;
- previous insulin;
- patient sensitivity estimate.

### Timing Assumptions

If CGM reports every five minutes, histories should reflect that cadence.

### Numerical Stability Concerns

Normalize using fixed clinical ranges or saved training statistics. Do not let near-zero features dominate with noise.

### Configuration Strategy

Observation version must be saved with checkpoint.

### Logging Strategy

Store observation schema in experiment artifacts.

### Testing Strategy

Test shape, dtype, bounds, and deterministic output.

---

## Action Space

### Architecture

Action spaces should have physical meaning.

Direct action examples:

- basal multiplier;
- basal delta U/h;
- correction bolus proposal.

Supervisory action examples:

- target glucose adjustment;
- aggressiveness;
- insulin cost multiplier;
- sensitivity multiplier;
- meal bolus fraction.

### APIs

Action mapper converts normalized RL action to physical command or controller parameters.

### State Definitions

Keep requested and delivered actions separate.

### Timing Assumptions

An action selected every 15 minutes should persist or be interpolated for lower-level control.

### Numerical Stability Concerns

Clip before applying and log clipping. Do not allow unbounded gain drift.

### Configuration Strategy

Action bounds in config and saved with model.

### Testing Strategy

Unit test every action bound and physical mapping.

---

## Replay and Training Pipeline

### Architecture

Use a trainer with callbacks. Do not put training loops inside agent classes.

### APIs

```python
Trainer.train(config) -> TrainingResult
Trainer.evaluate(checkpoint, eval_config) -> EvaluationReport
```

### State Definitions

Training state includes episode counters, optimizer state, RNG state, best checkpoint metadata, and evaluation history.

### Timing Assumptions

Evaluation should run at fixed intervals using fixed seeds.

### Numerical Stability Concerns

Use gradient clipping. Monitor NaNs in observations, rewards, losses, and actions.

### Configuration Strategy

All hyperparameters in YAML. No hidden preset logic.

### Logging Strategy

Log reward components and clinical metrics separately.

### Testing Strategy

Two-episode smoke tests and seeded reproducibility tests.

---

## Experiment Tracking

### Architecture

Start with local run directories. Add MLflow or Weights & Biases later.

### APIs

```python
ExperimentLogger.log_config(...)
ExperimentLogger.log_step(...)
ExperimentLogger.log_metrics(...)
ExperimentLogger.save_artifact(...)
```

### State Definitions

Run identity includes timestamp, experiment name, seed, git commit, config hash, and code version.

### Timing Assumptions

Flush logs at episode boundaries and evaluation boundaries.

### Numerical Stability Concerns

Detect and mark failed runs instead of silently producing partial metrics.

### Configuration Strategy

Each run directory contains the resolved config.

### Testing Strategy

Test run directory creation, metric writes, and artifact paths.

---

## Metrics Engine

### Architecture

Metrics are pure functions over trajectories and step records.

### APIs

```python
compute_clinical_metrics(records) -> dict
compute_safety_metrics(records) -> dict
compute_meal_response_metrics(records) -> dict
```

### State Definitions

Metrics should distinguish true glucose from sensor glucose.

### Timing Assumptions

When reporting clinical CGM metrics, state sampling frequency.

### Numerical Stability Concerns

Handle empty trajectories and missing data explicitly.

### Configuration Strategy

Metric thresholds in one constants file.

### Testing Strategy

Synthetic traces with known expected outputs.

---

## Evaluation Harness

### Architecture

Evaluation runs controllers over paired patient/scenario suites.

### APIs

```python
EvaluationHarness.run(controller_specs, scenario_suite, patient_suite) -> Report
```

### State Definitions

Report includes per-run metrics, aggregate metrics, confidence intervals, and failure cases.

### Timing Assumptions

Use the same horizon for compared controllers.

### Numerical Stability Concerns

Failed simulations should be counted and reported.

### Configuration Strategy

Evaluation suite YAML should specify patient/scenario seeds.

### Testing Strategy

Test paired comparison uses identical scenarios for all controllers.

---

## UI and Visualization

### Architecture

Backend API plus thin UI.

### APIs

FastAPI endpoints return typed schemas. UI consumes records and metrics.

### State Definitions

UI state includes selected profile, scenario, controller set, playback time, and comparison mode.

### Timing Assumptions

Playback should not rerun simulation on every slider move.

### Threading Considerations

Long simulations should run in a background task or stream steps.

### Numerical Stability Concerns

UI must handle failed simulations and NaNs gracefully.

### Configuration Strategy

UI reads available profiles, scenarios, and controllers from backend.

### Logging Strategy

Backend logs request config and run ID.

### Testing Strategy

API tests first; browser tests after UI stabilizes.

---

## Scenario Generation

### Architecture

Scenario generation should be deterministic from seeds and separate from the simulator.

### APIs

```python
ScenarioSampler.sample(seed, patient_profile) -> Scenario
```

### State Definitions

Scenario includes actual and announced events.

### Timing Assumptions

All generated events align to simulator output interval.

### Numerical Stability Concerns

Reject impossible scenarios.

### Configuration Strategy

Use distributions in YAML.

### Testing Strategy

Same seed produces same scenario.

---

## Config System

### Architecture

Keep current YAML loader concepts, add typed validation.

### APIs

`load_config(path, schema) -> ConfigObject`.

### State Definitions

Separate configs:

- patient;
- scenario;
- controller;
- environment;
- training;
- evaluation;
- UI.

### Timing Assumptions

All dt values explicit.

### Numerical Stability Concerns

Validate ranges for physiological and control parameters.

### Logging Strategy

Log resolved config, not just overrides.

### Testing Strategy

Invalid configs fail with clear errors.

---

## Model Checkpointing

### Architecture

Use framework-specific checkpoint files but framework-neutral metadata.

### APIs

```python
save_checkpoint(policy, metadata, path)
load_checkpoint(path) -> Policy
```

### State Definitions

Metadata includes observation schema, action schema, reward version, training config, evaluation metrics, and git commit.

### Timing Assumptions

Checkpoint only after evaluation or at safe intervals.

### Numerical Stability Concerns

Reject checkpoint if action/observation schema mismatch.

### Configuration Strategy

Checkpoint directory comes from config.

### Testing Strategy

Save/load round-trip produces same deterministic action.

---

## Debugging Tooling

### Architecture

Create CLI and visualization helpers for trajectory inspection.

### APIs

- `ap-rl-simulate`
- `ap-rl-evaluate`
- `ap-rl-compare`
- `ap-rl-debug-step`

### State Definitions

Debug output should include compartments, event list, controller metadata, safety decision, and reward components.

### Timing Assumptions

Step debugger should replay a saved trajectory deterministically.

### Numerical Stability Concerns

Add assertions for NaN, negative compartments, and impossible commands.

### Configuration Strategy

Debug CLI accepts run directory or config path.

### Testing Strategy

CLI smoke tests.

---

# 5. Safety and Clinical Logic

## Hypoglycemia Prevention

Hypoglycemia prevention must be implemented as hard logic, not as a reward preference.

Recommended layers:

1. Reactive severe low:
   - If glucose <54 mg/dL, suspend all insulin and log emergency safety event.
2. Reactive low:
   - If glucose <70 mg/dL, suspend basal and block bolus/tail bolus.
3. Falling near-low:
   - If glucose <80 mg/dL and rate < -1 mg/dL/min, suspend or reduce strongly.
4. Predictive low:
   - Predict 30 minutes ahead using linear trend initially.
   - If predicted glucose <70 mg/dL, suspend.
   - If predicted glucose <80 mg/dL, reduce basal.
5. IOB-aware low:
   - Estimate glucose drop from IOB and ISF.
   - If `current_glucose - iob * isf < 80`, reduce or suspend.

## Insulin Hard Limits

Initial conservative defaults:

```yaml
safety:
  max_basal_u_h: 5.0
  max_bolus_u: 15.0
  max_iob_u: 20.0
  min_bolus_interval_min: 15
  max_daily_insulin_u: 100.0
  max_rate_change_u_h_per_min: 2.0
  suspend_glucose_mgdl: 70.0
  severe_hypo_glucose_mgdl: 54.0
  reduce_glucose_mgdl: 90.0
  predictive_horizon_min: 30
  predictive_suspend_threshold_mgdl: 70.0
  predictive_reduce_threshold_mgdl: 80.0
```

These are not final clinical parameters. They are research-simulator safety bounds and should be configurable per patient profile.

## IOB Handling

The current exponential update:

```python
iob = iob * 0.985 + delivered / 60.0
```

should be removed from primary code. It underestimates active insulin during the most dangerous post-delivery period.

Preferred implementation:

1. Compute IOB from Hovorka insulin compartments.
2. Validate against a rapid-acting insulin action curve.
3. Expose both raw compartment values and normalized IOB.
4. Use IOB in safety, observations, metrics, and UI.

## Prediction Horizons

Use multiple horizons:

- 30 minutes: safety suspend.
- 60 minutes: near-term risk.
- 180 minutes: MPC insulin planning.
- 24 hours: evaluation only, not controller prediction.

## Uncertainty Handling

Initial uncertainty logic:

- If sensor reading is missing, stale, or implausible, switch to conservative mode.
- If glucose prediction uncertainty is high, widen safety margins.
- If ensemble or transformer forecast is later added, use prediction intervals for risk-aware control.

## Fallback Controllers

Fallback chain:

```text
Hybrid MPC+RL
  -> MPC only
  -> safe PID
  -> basal-only
  -> suspend
```

Fallback triggers:

- RL checkpoint missing.
- RL action NaN or out-of-schema.
- MPC solver timeout.
- MPC infeasible.
- Sensor dropout.
- Simulator detects NaN.
- Safety constraint violation rate exceeds threshold.

## Action Clipping

Clip in two stages:

1. Environment or controller clips requested action to declared physical range.
2. Safety supervisor clips delivered command to patient safety range.

Always log both requested and delivered values. Do not train only on requested insulin if delivered insulin differs; the agent must see safety-modified outcomes.

## Emergency Logic

Emergency states:

- severe hypoglycemia;
- glucose below 70 and falling;
- excessive IOB;
- repeated unsafe bolus request;
- numerical instability;
- impossible sensor reading;
- pump failure scenario.

Emergency actions:

- suspend insulin;
- freeze RL/MPC adaptation;
- switch to fallback;
- log event;
- continue simulation if safe for evaluation, or terminate if configured.

## Safety Validation Methods

Safety must pass:

- deterministic unit tests;
- closed-loop stress tests;
- Monte Carlo patient tests;
- adversarial scenario tests;
- regression tests for known crash cases;
- comparison of requested vs delivered insulin.

## Required Safety Stress Tests

Minimum stress suite:

- 100 g meal without announcement.
- Double meal bolus.
- Late-night correction bolus.
- Exercise during insulin peak.
- Sensor dropout for 60 minutes.
- Sensor reads falsely high.
- Sensor reads falsely low.
- Pump occlusion with delayed recovery.
- Insulin-sensitive patient with normal meal.
- Insulin-resistant patient with heavy meal.
- Severe dawn phenomenon.
- Carb count error: reported 30 g, actual 60 g.
- Alcohol-like delayed hypoglycemia scenario.
- Sick-day elevated EGP scenario.

Pass/fail criteria:

- Time below 54 mg/dL should be under 1% in standard suites.
- Any severe hypo in stress suite must be reported and reviewed.
- No controller may deliver insulin while glucose is below suspend threshold.
- No bolus may be delivered during active lockout.
- No total daily insulin above configured cap.

---

# 6. RL/Control Strategy Recommendation

## Current RL Approach

The current approach uses A2C to adjust PID gains. It should not remain the main architecture.

Technical problems:

- Action does not directly represent insulin.
- Action effect is delayed and indirect.
- PID gains drift over time.
- The plant changes as gains change, making learning non-stationary.
- PID internal state is part of the effective state, but the formulation is not cleanly Markov.
- Reward conflicts with controller setpoint.
- Safety is incomplete.
- A2C implementation has fixed policy std and duplicated training logic.

The current approach can be preserved only as a legacy baseline for comparison.

## Should RL Remain Primary?

No. RL should not be the primary insulin controller in this project.

Reasons:

- Insulin delivery is safety-critical.
- The delay between action and glucose effect is long.
- Pure RL requires large scenario diversity and robust safety constraints.
- Classical and model-based controllers are more interpretable.
- MPC naturally handles prediction and constraints.
- RL is valuable for adaptation, not as the first line of safety-critical dosing.

## PID

PID is useful as:

- a simple baseline;
- a fallback controller;
- a teaching comparison.

PID is weak because:

- insulin action delay causes oscillation;
- IOB is not naturally handled;
- meal disturbances require feedforward bolus;
- derivative term is noise-sensitive;
- gain scheduling requires analysis.

Recommendation:

- Keep safe PID baseline.
- Do not tune PID gains every minute with RL.

## MPC

MPC should be the primary controller baseline.

Strengths:

- Handles delayed insulin dynamics through prediction horizon.
- Supports hard constraints.
- Produces interpretable plans.
- Works without training.
- Provides expert demonstrations for imitation learning.

Weaknesses:

- Requires a prediction model.
- Can be computationally expensive.
- Can fail if model mismatch is large.

Recommendation:

- Implement MPC after simulator and safety are stable.
- Compare every RL result against MPC.

## Hybrid MPC+RL

Hybrid MPC+RL is the recommended final direction.

MPC handles:

- basal planning;
- delayed insulin action;
- constraints;
- immediate optimization.

RL handles:

- adapting target within safe range;
- adjusting aggressiveness;
- estimating sensitivity changes;
- selecting conservative vs aggressive mode;
- tuning meal bolus fraction.

This is more defensible than direct RL because the learned part is slower, bounded, and interpretable.

## Ensemble Systems

Ensembles help mainly with uncertainty and patient identification.

Recommended uses:

- ensemble MPC over different patient parameters;
- ensemble forecasters for glucose prediction intervals;
- uncertainty-triggered conservative fallback.

Do not use ensembles to hide a weak base controller. Use them after the base MPC is correct.

## Model-Based RL

Model-based RL is promising but should be a later research extension.

Possible paths:

- differentiable Hovorka simulator;
- transformer forecaster inside MPC;
- Dreamer-style latent dynamics;
- offline model-based planning.

Recommendation:

- Do not start here.
- First build deterministic simulator, safety, MPC, and evaluation.

## Final Control Recommendation

Order of controller development:

1. Basal-only fallback.
2. Safe discrete PID baseline.
3. MPC primary controller.
4. Ensemble MPC optional robustness.
5. RL supervisor over MPC.
6. Safe RL constraints and uncertainty-aware extensions.
7. World models or transformer forecasting as research upgrades.

---

# 7. Testing and Validation Framework

## Unit Tests

Required unit tests:

- Hovorka RHS equations.
- RK4 integrator.
- Circadian EGP.
- Meal event handling.
- Exercise sensitivity.
- Insulin calculator.
- IOB model.
- Safety supervisor.
- PID controller.
- Reward function.
- Observation builder.
- Action mapper.
- Metrics.
- Scenario schema.
- Config validation.

## Integration Tests

Required integration tests:

- basal-only closed loop;
- PID closed loop;
- MPC closed loop;
- hybrid controller smoke;
- simulation runner reproducibility;
- evaluation harness paired comparison;
- training smoke test.

## Deterministic Replay Tests

Every important simulation should be reproducible from:

- config;
- seed;
- controller checkpoint;
- scenario ID;
- patient parameters;
- code version.

Replay test:

1. Run simulation.
2. Save records.
3. Re-run with same inputs.
4. Compare glucose and insulin trajectories.

## Golden Trajectory Tests

Golden trajectories should include:

- fasting basal-only day;
- single meal no insulin;
- meal plus bolus;
- exercise event;
- circadian EGP day;
- safety suspend case.

Use golden tests to detect silent physics changes.

## Monte Carlo Scenario Testing

Run population evaluation:

- at least 50 patients x 20 scenarios for development;
- at least 100 patients x 50 scenarios for research claims.

Report:

- mean;
- median;
- standard deviation;
- confidence intervals;
- worst-case;
- severe safety failures.

## Robustness Testing

Robustness dimensions:

- patient sensitivity;
- meal timing;
- carb counting error;
- exercise intensity;
- sensor noise;
- sensor delay;
- insulin absorption variability;
- pump delivery failure.

## Adversarial Scenarios

Adversarial suite should find failure modes, not just validate happy paths.

Examples:

- meal during falling glucose;
- exercise during peak IOB;
- sensor falsely high causing over-delivery;
- missed meal announcement;
- repeated correction requests;
- high-fat delayed absorption meal.

## Regression Testing

Every bug from `CodeReview/issues.json` should map to a regression test or an explicit migration decision.

Critical regressions:

- BUG-001: integrator mismatch.
- BUG-002: PID-delta action space removed from primary env.
- BUG-003: reward/TIR anti-correlation.
- BUG-005: temp file race.
- BUG-006: IOB model.
- BUG-007/022: total insulin safety.
- BUG-008/021: PID wall-clock time.
- BUG-017: `t_max_G` inconsistency.

## Reproducibility Validation

Required checks:

- no global RNG in training loops;
- resolved config saved;
- checkpoint metadata saved;
- package versions saved;
- seeds saved;
- evaluation scenario IDs saved.

## Clinical Metrics and Thresholds

Initial standard targets:

- TIR 70-180 mg/dL: target >70%.
- TBR 54-70 mg/dL: target <4%.
- TBR <54 mg/dL: target <1%.
- TAR 180-250 mg/dL: target <25%.
- TAR >250 mg/dL: target <5%.
- CV: target <36%.

For this research simulator, failing a threshold does not mean clinical failure, but it means the controller should not be presented as improved without explaining the failure.

---

# 8. UI Rewrite Plan

## Why the Current UI Fails

The current `app/app.py` is useful as a prototype but not as a maintainable UI.

Problems:

- It directly instantiates `DiabetesPIDEnv`.
- It monkey-patches `env._skip_reload`.
- It accesses `env.patient.time`.
- It depends on Streamlit reruns.
- It mixes layout, plotting, state, simulation construction, and checkpoint loading.
- It cannot stream or play simulation naturally.
- It lacks robust error boundaries.
- It labels the simulator like a digital twin even though no real patient calibration exists.
- It cannot show controller internals well.

## Target Architecture

Backend:

- FastAPI service.
- Simulation API.
- Streaming API.
- Evaluation API.
- Typed request/response schemas.

Frontend:

- Streamlit thin client initially.
- React optional later.
- No low-level simulator imports.

## State Management

UI state:

- selected patient;
- selected scenario;
- selected controller(s);
- simulation run ID;
- playback time;
- chart options;
- selected debug overlays.

Simulation state lives in backend result objects, not in Streamlit global execution.

## Playback Engine

Required controls:

- run;
- reset;
- play;
- pause;
- step one minute;
- step five minutes;
- speed control;
- scrubber.

Playback should advance through existing records. It should not rerun simulation on every UI interaction.

## Chart Synchronization

Charts must share the same time axis:

- glucose;
- basal and bolus insulin;
- total insulin;
- IOB;
- COB;
- meals;
- exercise;
- safety events;
- controller predictions.

Hover on one chart should align with all charts.

## Streaming Updates

Use server-sent events or WebSocket later. Start with batch simulation and playback. Add streaming after backend contracts are stable.

## Scenario Editing

Scenario editor should validate:

- time range;
- carb range;
- exercise duration;
- duplicate events;
- impossible values.

It should show actual vs announced carbs when carb-counting error is enabled.

## Experiment Comparison

Comparison mode:

- same patient;
- same scenario;
- same seed;
- multiple controllers;
- metrics table;
- overlay charts;
- safety event comparison.

## Debugging Overlays

Include:

- requested vs delivered insulin;
- safety active constraints;
- predicted glucose horizon;
- MPC planned basal;
- RL supervisor action;
- IOB curve;
- sensor vs true glucose.

## Acceptance Criteria

- UI does not import legacy `DiabetesPIDEnv`.
- UI can compare PID, MPC, and hybrid.
- UI can replay without rerunning.
- Errors are shown as readable messages.
- Missing checkpoint does not imply RL results.

---

# 9. File and Folder Migration Plan

## Ideal Directory Tree

```text
src/ap_rl/
  core/
    __init__.py
    types.py
    constants.py
    units.py
    config.py

  simulation/
    __init__.py
    simulator.py
    patient.py
    hovorka.py
    integrators.py
    insulin_pk.py
    meal_model.py
    exercise.py
    sensor.py
    pump.py
    scenario.py
    population.py

  controllers/
    __init__.py
    base.py
    pid.py
    mpc.py
    rl_supervisor.py
    hybrid.py
    oracle.py
    safety.py

  environments/
    __init__.py
    glucose_control_env.py
    supervisory_env.py
    wrappers.py

  rewards/
    __init__.py
    clinical.py

  agents/
    __init__.py
    ppo.py
    sac.py
    networks.py
    replay.py

  training/
    __init__.py
    trainer.py
    experiment.py
    curriculum.py
    imitation.py
    callbacks.py

  evaluation/
    __init__.py
    metrics.py
    harness.py
    monte_carlo.py
    stress_tests.py
    comparison.py
    golden.py

  runtime/
    __init__.py
    records.py
    rollout.py

  visualization/
    __init__.py
    glucose_plot.py
    insulin_plot.py
    agp_report.py
    controller_viz.py
    comparison.py

  app/
    __init__.py
    server.py
    schemas.py
    streamlit_app.py
    components/
```

## Migration Sequence

1. Add new modules without deleting old ones.
2. Move Hovorka equations into `simulation/hovorka.py`.
3. Add safety supervisor.
4. Add controller interface and safe PID.
5. Add simulation runner.
6. Add new Gymnasium envs.
7. Move metrics to expanded engine.
8. Add evaluation harness.
9. Add MPC.
10. Add training infrastructure.
11. Add hybrid RL.
12. Rewrite UI.
13. Delete legacy modules after compatibility tests pass.

## Deprecated Modules

Deprecate:

- `src/ap_rl/envs/diabetes_pid_env.py`
- `src/ap_rl/envs/hovorka_gym_env.py`
- `src/ap_rl/utils/pid_controller.py`
- old TensorFlow A2C modules if PyTorch migration is accepted.

## Modules to Delete

Delete after migration:

- `docs/legacy-pid-tuner/`
- `TestCaseManager/`
- `verify_best_model.py`
- temp-file writing path and `atexit` cleanup.

## Modules to Preserve

Preserve:

- `src/ap_rl/utils/insulin_calculator.py`
- `src/ap_rl/utils/config.py`
- `src/ap_rl/utils/paths.py`
- `src/ap_rl/utils/seed.py`
- `src/ap_rl/envs/profile_loader.py`
- `src/ap_rl/envs/scenario_builder.py`
- `src/ap_rl/envs/defaults.py`
- `src/ap_rl/runtime/rollout.py` concepts.
- `src/ap_rl/visualization/publication.py`.

## Naming Conventions

- Use physical units in names: `_mgdl`, `_u_h`, `_u`, `_min`, `_g`.
- Use `Config` suffix for config dataclasses.
- Use `Record` suffix for logged step/episode data.
- Use `Controller` suffix for control policies.
- Use `Model` suffix for physiological or learned dynamics models.

## Migration Notes

- Do not change public behavior and architecture in the same commit when avoidable.
- Keep compatibility wrappers for one or two phases.
- Every migration step needs tests before deleting old modules.
- Checkpoints trained with old observations/actions must be marked legacy.

---

# 10. Research-Grade Improvements

These are optional after the foundation is stable. Do not start them before simulator, safety, evaluation, and MPC are working.

## Safe RL

- Complexity: hard.
- Value: critical.
- Feasibility: good after supervisory RL exists.
- Demo impact: high.
- Publication potential: high.

Implement constrained MDP training where severe hypo is a cost constraint, not just a reward penalty. Use Lagrangian PPO/RCPO before attempting full CPO.

## Uncertainty-Aware Control

- Complexity: hard.
- Value: high.
- Feasibility: medium.
- Demo impact: very high.
- Publication potential: high.

Use ensembles of forecasters or supervisors. Fall back to conservative MPC when prediction uncertainty is high.

## Ensemble MPC

- Complexity: medium.
- Value: high.
- Feasibility: high.
- Demo impact: high.
- Publication potential: good for biomedical venues.

Run multiple MPC models with different patient parameters and weight them by recent prediction error.

## Transformer Glucose Forecasting

- Complexity: medium.
- Value: high.
- Feasibility: medium after data generation pipeline exists.
- Demo impact: very high.
- Publication potential: high.

Train multi-horizon forecasts at 30/60/90/120 minutes from glucose, insulin, carbs, exercise, and time features. Use attention visualization for explainability.

## Differentiable Simulator

- Complexity: medium.
- Value: high.
- Feasibility: medium.
- Demo impact: high.
- Publication potential: workshop/conference.

Reimplement Hovorka in PyTorch or JAX to optimize controller parameters through the simulator. Useful for model-based initialization, not first foundation step.

## Latent World Models

- Complexity: very hard.
- Value: high.
- Feasibility: low until data/evaluation are mature.
- Demo impact: very high.
- Publication potential: top venue if successful.

Use Dreamer-style latent dynamics. Requires strong sequence modeling and long insulin delay handling.

## Digital Twins

- Complexity: hard.
- Value: high.
- Feasibility: medium.
- Demo impact: very high.
- Publication potential: journal.

Only use the term digital twin after there is real or simulated patient-specific calibration and uncertainty estimates.

## Meta-Learning

- Complexity: very hard.
- Value: high.
- Feasibility: medium after population sampler exists.
- Demo impact: high.
- Publication potential: top venue.

Learn patient embeddings or use MAML/PEARL-like adaptation.

## Offline RL

- Complexity: medium.
- Value: high.
- Feasibility: high after trajectory logging exists.
- Demo impact: high.
- Publication potential: conference.

Generate logged trajectories from PID, MPC, and hybrid controllers. Train CQL/IQL/Decision Transformer policies without online unsafe exploration.

## Imitation Learning

- Complexity: easy-medium.
- Value: high.
- Feasibility: high after MPC exists.
- Demo impact: high.
- Publication potential: good if expert data are credible.

Use MPC as an expert for behavioral cloning before RL fine-tuning.

## Diffusion Scenario Generation

- Complexity: hard.
- Value: medium.
- Feasibility: medium.
- Demo impact: high.
- Publication potential: workshop/conference.

Generate diverse meal/exercise/stress sequences after baseline scenario schema exists.

## Explainable AI

- Complexity: easy-medium.
- Value: medium.
- Feasibility: high.
- Demo impact: very high.
- Publication potential: workshop.

Implement counterfactuals and feature attribution for supervisor decisions and forecasts.

---

# 11. Final Recommended Execution Order

This is the master checklist. Execute in this order.

## Foundation

1. Record current test and rollout baseline.
2. Create regression fixture directory for golden trajectories.
3. Add `core/types.py` with shared dataclasses.
4. Add `simulation/integrators.py` with fixed-step RK4.
5. Move Hovorka RHS into `simulation/hovorka.py`.
6. Remove Numba/Python physics divergence.
7. Include circadian EGP in the single authoritative integrator.
8. Remove hidden Hovorka parameter defaults from ODE methods.
9. Add in-memory meal/exercise APIs.
10. Remove exact floating-point event matching.
11. Add Hovorka reference tests against RK45.
12. Add golden trajectory tests.

## Safety

13. Add `controllers/safety.py`.
14. Define insulin command and safety decision types.
15. Implement total insulin command clamping.
16. Add bolus lockout enforcement.
17. Add IOB cap enforcement.
18. Replace exponential IOB with compartment or validated PK IOB.
19. Add reactive low suspend.
20. Add predictive low suspend.
21. Add safety event logging.
22. Add safety unit and property tests.

## Controllers and Runner

23. Add controller base protocol.
24. Implement basal-only fallback.
25. Implement safe discrete-time PID.
26. Remove wall-clock time from PID path.
27. Add derivative filtering and anti-windup.
28. Add simulation runner producing step records.
29. Refactor rollout to use new runner.
30. Remove TensorFlow top-level import from baseline rollout.
31. Add deterministic runner tests.

## Environments and Rewards

32. Add Gymnasium `GlucoseControlEnv`.
33. Add Gymnasium `SupervisoryEnv`.
34. Remove PID-delta action space from primary training.
35. Define observation builders and schemas.
36. Define physical action mappers.
37. Add bounded clinical reward.
38. Remove catastrophic reward spikes from primary reward.
39. Add observation/action/reward tests.
40. Run Gymnasium checker.

## Evaluation

41. Expand clinical metrics.
42. Add safety metrics.
43. Add evaluation harness.
44. Add fixed smoke, nominal, held-out, and stress suites.
45. Add Monte Carlo evaluator.
46. Add paired controller comparison.
47. Add evaluation CLI.
48. Make checkpoint selection use clinical metrics.

## Training

49. Add trainer abstraction.
50. Remove monkey-patched `agent.train`.
51. Replace global RNG usage with seeded generators.
52. Add experiment run directory format.
53. Add config dumping.
54. Add checkpoint metadata.
55. Add training smoke test.
56. Decide PyTorch/SB3/CleanRL migration.
57. Mark TensorFlow A2C as legacy if migrating.

## MPC

58. Implement MPC controller.
59. Add MPC config.
60. Add MPC solver fallback.
61. Log MPC prediction and plan metadata.
62. Evaluate MPC against basal-only and safe PID.
63. Tune MPC for safety before TIR.
64. Add MPC tests.

## Hybrid RL

65. Implement RL supervisor action schema.
66. Implement hybrid controller.
67. Train supervisor over MPC, not PID gains.
68. Pretrain from MPC demonstrations if useful.
69. Evaluate against MPC-only on paired scenarios.
70. Reject hybrid checkpoints that worsen severe hypo.

## Scenario and Population

71. Add structured scenario schema.
72. Add patient population sampler.
73. Add stress scenario catalog.
74. Add deterministic scenario sampling.
75. Add scenario validation tests.

## UI

76. Add FastAPI backend schemas.
77. Add `/simulate` endpoint.
78. Add `/simulate/stream` endpoint.
79. Rebuild Streamlit as thin client.
80. Remove `env._skip_reload` monkey-patch.
81. Add playback without rerun.
82. Add synchronized charts.
83. Add controller comparison mode.
84. Add safety/debug overlays.
85. Add scenario editor validation.

## Cleanup

86. Delete `docs/legacy-pid-tuner/`.
87. Delete `TestCaseManager/`.
88. Delete `verify_best_model.py`.
89. Delete temp-file reset path.
90. Delete `atexit` cleanup.
91. Delete or archive `hovorka_gym_env.py`.
92. Move legacy PID-delta environment under a clearly marked legacy namespace if it must remain for old checkpoint replay.
93. Update README with new architecture and research/demo disclaimer.
94. Add architecture docs.
95. Run full tests and evaluation.
96. Tag the first stable research-platform version.

## Final Acceptance Criteria

The rebuild is acceptable when:

- Simulator physics do not depend on optional Numba availability.
- No reset or step writes temp files.
- Safety constrains total insulin, including bolus and tail delivery.
- IOB is physiologically meaningful.
- PID-delta RL is no longer primary.
- MPC baseline exists and is evaluated.
- RL supervisor is optional and bounded.
- Checkpoints are selected by clinical metrics.
- Evaluation reports TIR, TBR, TAR, CV, GMI, risk indices, insulin, and safety events.
- Same seed and config reproduce the same trajectory.
- UI consumes backend records and does not monkey-patch simulator internals.
- Legacy code is either deleted or clearly isolated.

