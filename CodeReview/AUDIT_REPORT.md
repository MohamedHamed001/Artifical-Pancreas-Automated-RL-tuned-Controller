# FULL SYSTEM AUDIT REPORT

## Artificial Pancreas RL-Tuned PID Controller

**Audit Date:** 2026-05-15
**Auditor Perspective:** RL Scientist / Control Systems Engineer / Software Architect
**Scope:** Complete repository — every source file, architecture decision, data flow, and scientific assumption

---

## 1. EXECUTIVE SUMMARY

### Overall Assessment: STRUCTURALLY FLAWED — REQUIRES FUNDAMENTAL REDESIGN

The system has a well-intentioned architecture but suffers from **compounding design errors** at every layer: the patient model uses inconsistent integrators, the RL formulation is scientifically unsound (tuning PID deltas is the wrong abstraction), the reward function has perverse incentives, the training pipeline lacks proper evaluation methodology, and the UI is a monolithic Streamlit script with no separation of concerns.

The most damaging single issue is the **fundamental RL formulation**: using A2C to output `(dKp, dKi, dKd)` deltas that adjust a PID controller's gains at every minute. This creates a **meta-controller problem** where the RL agent must learn a non-stationary control law through a noisy, delayed intermediary (the PID), rather than directly learning the insulin dosing policy. This is like teaching someone to drive by having them adjust the steering sensitivity every second rather than turning the wheel.

### Root Causes of Failure

1. **Wrong RL formulation**: The action space (PID gain deltas) is an indirect, poorly-conditioned mapping to insulin delivery. The agent fights the PID dynamics instead of learning insulin control directly.

2. **Simulator inconsistency**: The Numba fast path uses **forward Euler** with dt=1min while the fallback uses **RK45 with adaptive stepping**. These produce different trajectories. The Numba path omits circadian EGP modulation present in the Python path.

3. **Reward function instability**: Rewards span ±10,000 per step. The zone model, rate penalties, IOB brakes, and bonuses interact chaotically. A single hypo event (-10,000) dominates hundreds of good steps (+50 each), creating an extremely sparse and noisy learning signal.

4. **No proper baseline comparison**: There is no well-tuned standalone PID or MPC baseline. The "baseline" is the unmodified PID with frozen gains, which is deliberately suboptimal to make RL look good — but RL can't beat it reliably either.

5. **Training on reward, not on clinical outcomes**: The best checkpoint is selected by mean rolling reward, not by clinical metrics like TIR. The `generalize` preset attempts TIR-based selection but the underlying reward still drives gradient updates.

### Critical Architectural Flaws

- **PID-delta action space** makes the MDP non-Markov (current gains are hidden state the agent must remember via observation features)
- **No insulin-on-board (IOB) pharmacokinetic model** — the IOB tracking uses a simplistic exponential decay (`0.985^t`) rather than actual insulin absorption curves
- **Forward Euler integration** (Numba path) is numerically unstable for stiff ODE systems at dt=1min
- **Temp file I/O on every episode reset** (`temp_data/meal_temp.data`) — unnecessary disk I/O that breaks parallelism and reproducibility
- **Global state pollution** via `atexit` handler that removes `temp_data/`

### Most Dangerous Issues

| # | Issue | Severity |
|---|-------|----------|
| 1 | Numba integrator diverges from Python integrator (different physics) | **CRITICAL** |
| 2 | Reward of -10,000 for glucose < 40 makes learning catastrophically unstable | **CRITICAL** |
| 3 | No hard insulin dose ceiling — agent can drive total insulin to 10+ U/h | **CRITICAL** |
| 4 | PID integral windup guard of 50.0 is enormous (allows massive overshoot) | **HIGH** |
| 5 | Forward Euler with dt=1 can produce negative glucose concentrations | **HIGH** |

### Most Salvageable Parts

- **HovorkaPatient ODE equations** (Python path) — scientifically correct Hovorka model, needs integration method fix only
- **InsulinCalculator** — sound clinical heuristics (500/1500 rules, split bolus, ISF-adaptive dosing)
- **Evaluation metrics** — `glucose_trajectory_summary` is clean and correct
- **Publication plotting** — well-structured matplotlib helpers
- **Config/YAML infrastructure** — clean, minimal, correct
- **Profile/scenario system** — good concept, needs more patient variety

---

## 2. ARCHITECTURE REVIEW

### Current Topology

```
User → Streamlit (app.py) → rollout.py → DiabetesPIDEnv
                                              ├── HovorkaPatient (ODE simulator)
                                              ├── PID controller
                                              ├── InsulinCalculator (bolus math)
                                              └── MealParser (scenario loader)

Training: train_a2c.py → DiabetesA2CAgent → DiabetesPIDEnv
                              ├── DiabetesActor (TF/Keras)
                              └── DiabetesCritic (TF/Keras)
```

### Dependency Graph

```
ap_rl.__init__
├── envs
│   ├── diabetes_pid_env.py  ← MONOLITH: 674 lines, 8 responsibilities
│   │   ├── hovorka_patient.py (ODE simulator)
│   │   ├── insulin_calculator.py (bolus math)
│   │   ├── meal_parser.py (file I/O)
│   │   └── pid_controller.py
│   ├── hovorka_gym_env.py (optional Gymnasium wrapper — barely functional)
│   ├── profile_loader.py (YAML patient profiles)
│   ├── scenario_builder.py (YAML meal schedules)
│   └── defaults.py (hardcoded Hovorka params)
├── agents
│   ├── diabetes_a2c_agent.py (training loop + agent orchestrator)
│   ├── diabetes_a2c_actor.py (TF policy network)
│   └── diabetes_a2c_critic.py (TF value network)
├── training
│   └── train_a2c.py (preset configs + multi-profile curriculum)
├── runtime
│   └── rollout.py (episode runner + state-dim auto-detection)
├── evaluation
│   └── metrics.py
├── visualization
│   └── publication.py
└── utils
    ├── config.py, paths.py, seed.py
    ├── pid_controller.py, insulin_calculator.py, meal_parser.py
    ├── checkpoint_filenames.py, hardware.py
    └── __init__.py
```

### Data Flow Issues

1. **Meal data writes to disk on every reset**: `DiabetesPIDEnv.reset()` writes `temp_data/meal_temp.data` and `temp_data/exercise_temp.data`, then `HovorkaPatient.load_meal_data()` reads them back. This is unnecessary — the data is already in memory as `self.meal_data`.

2. **Observation noise applied inconsistently**: `_observe_glucose()` adds Gaussian noise, but this noisy value is only used in `_get_state()`. The reward function at line 601 uses `new_glucose` (ground truth from ODE), creating an information asymmetry where the agent sees noisy observations but is rewarded on clean signals.

3. **PID time basis uses `time.time()`**: The vendored PID controller defaults to `time.time()` for delta_time calculation. The env passes `current_time` manually, but the PID's `__init__` still calls `time.time()` creating a wall-clock dependency for the first step.

### Control Loop Structure

```
Per minute (env.step):
  1. Agent outputs (dKp, dKi, dKd) ∈ [-0.1, 0.1]³
  2. Gains scaled: dKp *= 0.2, dKi *= 0.1, dKd *= 0.1
  3. Gains clipped: Kp ∈ [0.01, 10], Ki ∈ [0, 1], Kd ∈ [0, 1]
  4. PID computes output from glucose error
  5. Basal rate = base_basal + pid_adjustment * 0.01
  6. Safety clamp applied
  7. Meal bolus added (if meal event)
  8. Total insulin = basal + bolus → Hovorka ODE step
  9. Reward computed from new glucose
```

**Critical problem**: The PID output is scaled by 0.01 at step 5, meaning the PID controller must produce outputs of order 100 to have a meaningful effect on basal rate. But the integral windup guard is ±50, so the PID can at most shift basal by ±0.5 U/h from its base rate. This makes the PID adjustment band extremely narrow — the agent's gain adjustments have almost no effect in practice.

### UI/Backend Coupling

The Streamlit app (`app/app.py`) at 450+ lines has tight coupling:
- Directly imports and instantiates `DiabetesPIDEnv`
- Builds env with `_skip_reload = True` monkey-patching
- Accesses `env.patient.time` for timestamps
- Uses `@st.cache_resource` for TensorFlow model caching (fragile)
- No WebSocket/API layer — everything is synchronous Streamlit reruns

---

## 3. RL ANALYSIS

### State Representation Issues

**19-D observation vector** (file: `diabetes_pid_env.py:305-366`):

| Dim | Feature | Normalisation | Problem |
|-----|---------|---------------|---------|
| 1 | glucose/400 | ✓ | Range [0, ~1.5] but physiological max is ~600 → clips at 1.5 |
| 2 | glucose_rate/100 | Poor | Typical rate is ±3 mg/dL/min → value is ±0.03, near-zero signal |
| 3 | error/200 | OK | |
| 4 | PID.ITerm/100 | Poor | ITerm windup guard is 50 → max 0.5, but typical values are <1 → near-zero |
| 5 | PID.DTerm/10 | Poor | DTerm is rate/delta_time — can spike to ±50 → saturates at ±5 |
| 6 | Kp/10 | OK | |
| 7 | Ki raw | **BAD** | Ki ∈ [0, 1] — not normalised, but ranges same as normalised features |
| 8 | Kd raw | **BAD** | Kd ∈ [0, 1] — same issue |
| 9-10 | time_since_meal/insulin | OK | Clipped to [0, 1] |
| 11 | exercise_status | OK | Binary 0/1 |
| 12-13 | circadian sin/cos | OK | |
| 14 | glucose_rate_2/100 | Same as dim 2 | Near-zero signal |
| 15 | bolus_remaining_norm | OK | |
| 16 | IOB/10 | Poor | Typical IOB is 0.1–0.5 U → value is 0.01–0.05, near-zero |
| 17-19 | ISF/CR/weight | OK | Normalised to ~[0, 1] |

**Core problem**: Features 2, 4, 5, 14, 16 are near-zero most of the time, providing negligible gradient signal. The agent effectively operates on ~12 useful dimensions out of 19.

### Action Space Issues

**The action space is fundamentally wrong.**

The agent outputs `(dKp, dKi, dKd) ∈ [-0.1, 0.1]³`, then scaled by `(0.2, 0.1, 0.1)` → effective deltas are `[-0.02, +0.02]` for Kp and `[-0.01, +0.01]` for Ki/Kd per step.

Problems:
1. **Indirect control**: The agent adjusts PID gains, which adjust PID output, which scales by 0.01, which adjusts basal rate. The insulin delivered is 3 levels of indirection from the action. This makes credit assignment nearly impossible.
2. **Gain drift**: Continuous small adjustments cause random-walk-like gain trajectories. After 1440 steps, Kp can drift from 0.5 to anywhere in [0.01, 10] through cumulative noise.
3. **Non-stationarity**: The effective plant dynamics change every step because the gains change, making the value function approximation extremely difficult.
4. **PID redundancy**: If the agent could learn to set optimal gains, it would converge to fixed values and the deltas would go to zero. The PID does the actual control — the RL layer is overhead.

### Reward Shaping Flaws

**File: `diabetes_pid_env.py:368-476`**

1. **Reward scale mismatch**: Zone A gives +50 (Gaussian centered at 100), while Zone C hypo gives up to -3,844 (for glucose=20: `(70-20)² * 1.5`). The catastrophic penalty at glucose<40 is -10,000. This 200:1 ratio between normal rewards and penalties means a single bad step wipes out 200 good steps, creating extremely high variance in episode returns.

2. **Perverse incentive at 100 mg/dL target**: The reward peaks at 100 mg/dL, but the PID setpoint is 120 mg/dL. This creates a tug-of-war: the PID drives toward 120, the reward pulls toward 100. The agent learns to fight its own controller.

3. **IOB brake creates oscillation**: The IOB brake penalizes `Kp > threshold` when `IOB > 1.5`. But after Kp is reduced (due to penalty), IOB drops, penalty disappears, Kp rises again → oscillation cycle.

4. **Recovery bonus is exploitable**: Getting +8 reward for falling from >200 mg/dL encourages the agent to first allow spikes then recover, rather than preventing spikes.

5. **Stability bonus conflicts with meal response**: The +8 reward for `|glucose_diff| <= 1.5` penalizes appropriate post-meal glucose excursions. After a 60g meal, glucose should rise — penalizing this rise teaches the agent to over-dose insulin pre-emptively.

### Training Instability Evidence

From `training_log.txt`:
- Episode 7: reward = **-23,228** (TIR 76.2%) — sudden collapse
- Episode 10: reward = **-93,841** (TIR 96.7%) — catastrophic despite good TIR!
- Episodes 1-6: rewards 48k-75k

This shows reward is **anti-correlated with clinical quality**: episode 10 has 96.7% TIR (excellent) but the worst reward. The IOB brake and rate penalties dominate, punishing the agent for *how* it achieves good control, not *what* it achieves.

### Exploration Problems

- Gaussian noise `N(0, σ)` with `σ` decaying from 0.2 to 0.04 is applied to gain deltas. Since deltas are already tiny (0.02 max), noise dominates the signal for the first ~200 episodes.
- No parameter-space exploration (e.g., NoisyNets)
- No curiosity or intrinsic motivation
- No hindsight experience replay

### Overfitting Risks

- Training and evaluation use the same 16 scenarios from `data/test_scenarios/`
- The held-out split (`_select_held_out_cases`) is good in concept but only 16 total scenarios exist
- No domain randomization of Hovorka parameters during training (except `allprofiles` preset)
- No observation augmentation

### Safety Violations

The `_safety_clamp` function (line 478-495) is minimal:
- Only suspends basal below 70 mg/dL — bolus insulin is NOT clamped
- No maximum total insulin rate constraint
- No constraint on insulin stacking (IOB)
- No predictive low glucose suspend (only reacts to current glucose and 1-min rate)
- The 10 U/h upper clamp on basal (line 580) is dangerously high — typical maximum is 2-5 U/h

---

## 4. CONTROL-THEORY ANALYSIS

### PID Logic Flaws

**File: `pid_controller.py`**

1. **PTerm initialized to 0.2** (line 39: `self.PTerm = 0.2`): This is a non-zero initial proportional output that biases the controller from the first step. Should be 0.0.

2. **No derivative filtering**: The DTerm is raw `(error - last_error) / dt`. Without a low-pass filter, measurement noise causes derivative kicks. With the Numba integrator's quantization noise, this produces large spikes.

3. **Integral windup guard is symmetric** at ±50 (line 43): For insulin dosing, negative ITerm (requesting insulin reduction) should have a tighter bound than positive (requesting increase), because you can always stop giving insulin but can't take it back.

4. **sample_time is set to 0.0** and never enforced: The `update()` method does not gate on sample_time, meaning it can be called at any frequency. The env calls it every minute, but there's no enforcement.

5. **No anti-windup on setpoint change**: When the setpoint changes (it doesn't in practice, always 120), the integral term retains its accumulated value, causing a bump.

### Unsafe Gain Scheduling

The RL agent's gain adjustments create an implicit gain schedule, but:
- **Kp range [0.01, 10]** is extremely wide. At Kp=10, a 10 mg/dL error produces an output of 100, scaled by 0.01 → 1 U/h adjustment. This is physiologically significant and can cause severe hypoglycemia.
- **Ki range [0, 1]** with windup guard of 50 allows integral accumulation of 50 units over sustained error, translating to 0.5 U/h sustained offset.
- **No gain stability analysis**: There's no Nyquist/Bode analysis to ensure the closed-loop system remains stable across the gain range.

### Oscillation Causes

The 45-90 minute insulin action delay (pharmacokinetic lag in Hovorka model) combined with:
1. Aggressive Kp (agent can set Kp up to 10)
2. No prediction of insulin-on-board effect
3. 1-minute control interval (too fast for the plant dynamics — insulin effect is 30+ minutes delayed)

This creates classic **delayed-feedback oscillation**: the controller reacts to current glucose, but the insulin it delivered 45 minutes ago is still being absorbed. The controller stacks more insulin, then glucose crashes, then it stops insulin, but previously stacked insulin continues absorbing → severe hypoglycemia → glucose rebounds → cycle repeats.

### Physiological Mismatch

The PID output mapping (`pid_adjustment = -self.pid.output * 0.01`) means:
- PID setpoint error of 20 mg/dL → proportional output = 0.5 × 20 = 10 → insulin adjustment = -10 × 0.01 = -0.1 U/h
- **The sign is inverted**: positive error (glucose below target) produces positive PID output, which when multiplied by -0.01 reduces basal rate. This is correct directionally but the magnitude is tiny.
- A 100 mg/dL error (severe hyperglycemia) only changes basal by 0.5 U/h — far too slow to respond to a real crisis.

---

## 5. SIMULATOR ANALYSIS

### Physiological Realism

The Hovorka model is a **validated minimal model** (Hovorka et al., 2004) used in many AP studies. The implementation is largely correct with these caveats:

1. **10-state system** covers: subcutaneous insulin (S1, S2), plasma insulin (I), insulin actions (x1-x3), glucose masses (Q1, Q2), gut absorption (D1, D2). This is adequate for research-grade simulation.

2. **Missing counterregulation**: No glucagon response, no cortisol/growth hormone dawn phenomenon (beyond the simple sinusoidal EGP modulation).

3. **Meal model is simplistic**: Instantaneous carb loading into D1 (`self.D1 += carbs_mmol`) at exact meal time. Real meals have variable gastric emptying (15-90 min) depending on meal composition.

### Numerical Integration: THE CRITICAL BUG

**Numba path** (lines 323-389): Forward Euler with dt=1 minute.
```python
S1 + dS1, S2 + dS2, ...  # Euler: y(t+1) = y(t) + f(y(t)) * dt
```

**Python path** (lines 294-302): RK45 adaptive stepper via `scipy.integrate.solve_ivp`.

**These produce different trajectories.** For the stiff insulin dynamics (k_e=0.138, time constants of 7 minutes), forward Euler at dt=1min is marginally stable. The RK45 path takes internal sub-steps for accuracy. Training with Numba and evaluating without (or vice versa) creates silent distribution shift.

**Additionally**: The Numba path **omits the circadian EGP modulation** (compare line 360: `EGP = EGP_0 - x3 * EGP_0` vs line 250-253 which includes the sinusoidal term). This means:
- Numba patients have flat endogenous glucose production
- Python patients have dawn-phenomenon-like glucose rises
- A policy trained with Numba will fail on Python patients at dawn/dusk

### Exercise Modeling

The exercise model is a simple sensitivity multiplier:
```python
F_sensitivity = 1 + (F_peak - 1) * (1 - exp(-K_rise * t_rise))  # during exercise
F_sensitivity = 1 + (F_peak - 1) * exp(-K_decay * t_decay)       # after exercise
```

This only affects glucose uptake (`U_g = x1 * Q1 * F_sensitivity`), not EGP or insulin sensitivity. Real exercise has complex effects including:
- Increased hepatic glucose output
- Enhanced insulin sensitivity lasting 24-48 hours
- Adrenaline-mediated glucose elevation during intense exercise
- Glycogen depletion effects

The current model is acceptable for a demo but oversimplified for research.

### Time Discretization

The 1-minute timestep is appropriate for insulin pump control (pumps adjust basal every 5 minutes, sensors report every 1-5 minutes). However, the Euler integrator needs sub-stepping to maintain accuracy — a 15-second or 30-second internal step with 1-minute output would be much more numerically stable.

---

## 6. SCENARIO ENGINE REVIEW

### Current Weaknesses

- Only **16 test cases** in `data/test_scenarios/`
- All scenarios appear to be 1440-minute (24-hour) simulations
- No multi-day scenarios (dawn phenomenon, weekend patterns)
- No sick-day scenarios (stress hormones, fever)
- No alcohol scenarios (delayed hypoglycemia)
- No sensor failure/calibration scenarios
- No pump occlusion/failure scenarios

### Missing Variability

- Patient parameters are fixed per profile (4 profiles: controlled, insulin_resistant, insulin_sensitive, unstable)
- No inter-day insulin sensitivity variation
- No intra-day insulin sensitivity variation (beyond simple sinusoidal EGP)
- No random meal timing jitter
- No partial/forgotten bolus scenarios
- No carb counting error scenarios (±30% is typical)

### Missing Stochasticity

- Sensor noise is optional and fixed std (no drift, no calibration error)
- No insulin absorption variability
- No site-dependent absorption differences
- No physiological parameter drift within an episode

---

## 7. METRICS REVIEW

### Correct Metrics
- `percent_in_band` — correctly computed
- `glucose_trajectory_summary` — clean implementation with proper band definitions

### Misleading Metrics
- **Total episode reward** (shown in training log) is meaningless as a clinical metric. Episodes with 96.7% TIR can have -93k reward while episodes with 83.3% TIR have +48k reward.
- **Mean glucose** without standard deviation or percentiles obscures dangerous variability
- **TIR 80-140** (tight band) is non-standard — the clinical standard is 70-180

### Missing Clinical Metrics (International Consensus 2019)
- **Time below range (TBR)**: <54 mg/dL (severe hypo) — most critical safety metric
- **Glucose Management Indicator (GMI)**: estimated HbA1c
- **Coefficient of Variation (CV)**: glycemic variability, target <36%
- **Mean Amplitude of Glycemic Excursions (MAGE)**
- **Low Blood Glucose Index (LBGI)** and **High Blood Glucose Index (HBGI)** (Kovatchev)
- **Area under the curve** for hypo/hyper excursions
- **Maximum glucose rate of change**
- **Time to return to range** after meal bolus

### Missing Safety Metrics
- Maximum insulin stacking (peak IOB)
- Number of insulin suspension events
- Duration of insulin suspension events
- Number of predicted low glucose events vs actual
- False alarm rate for safety system

---

## 8. UI/UX AUDIT

### Why Current UI Fails

1. **No real-time playback**: The slider is a post-hoc scrubber, not a live animation. Users can't see the simulation evolve, reducing educational value.

2. **Monolithic 450-line file**: All plotting, state management, env construction, and UI layout in one file. No component reuse, no state machine.

3. **Streamlit rerun architecture**: Every slider change reruns the entire script. The `@st.cache_resource` prevents re-loading the actor, but the rollout itself re-executes on any UI interaction.

4. **No error boundaries**: If the simulation crashes (glucose < 0), the entire app crashes with a Python traceback.

5. **Misleading "Digital Twin" label**: The system is a simple forward simulation, not a digital twin (which implies real-time calibration against actual patient data).

6. **No controller introspection**: Can't see what the RL agent is "thinking" — no attention maps, no action explanations, no uncertainty estimates.

### Performance Issues
- Full 1440-step rollout takes 2-10 seconds (depending on Numba availability)
- Plotly renders 3 large figures per rerun
- No progressive rendering or streaming updates

---

## 9. SOFTWARE ENGINEERING AUDIT

### Code Smells

1. **God class**: `DiabetesPIDEnv` (674 lines) handles: ODE stepping, PID control, reward computation, bolus management, state observation, meal scheduling, scenario loading, file I/O, statistics, rendering. Should be at least 5 separate classes.

2. **Monkey-patching**: `env._skip_reload = True` used in app.py, rollout.py, and tests. Should be a constructor parameter.

3. **`atexit` handler** at module level in `diabetes_pid_env.py:660-673` — deletes `temp_data/` directory. This is a side effect of importing the module.

4. **Temp file I/O loop**: `reset()` → write files → `HovorkaPatient.load_meal_data()` → read files. The data is already in `self.meal_data`. This loop should be eliminated.

5. **Duplicated logic**: `_parse_data_file` in DiabetesPIDEnv partially duplicates `MealParser.parse_test_case` and `HovorkaPatient.load_meal_data`.

6. **Magic numbers**: `0.985` (IOB decay), `0.01` (PID output scaling), `0.55` (TDI factor), `18.0182` (mmol→mg/dL conversion) scattered through env without named constants.

### Anti-Patterns

1. **Training loop in agent class**: `DiabetesA2CAgent.train()` is a 160-line method that mixes training logic, logging, checkpointing, and evaluation. The `allprofiles` preset monkey-patches this method entirely.

2. **Hardcoded dimensions**: `observation_space = 19`, `action_space = 3`, `action_bound = 0.1` are hardcoded in multiple places (env, agent, rollout) rather than derived from a single source of truth.

3. **Import-time TensorFlow**: `agents/__init__.py` imports `DiabetesA2CAgent` which imports TF at the top level. Any code that imports the agents package pays the 5-10 second TF initialization cost.

### Missing Testing

- **No integration tests**: No test runs a full training loop (even for 2 episodes)
- **No regression tests**: No golden-file tests comparing simulation output against known-good trajectories
- **No property tests**: No hypothesis-based testing of ODE stability, reward bounds, etc.
- **No performance tests**: No benchmarks for simulation speed, training throughput
- **test_insulin_calculator.py line 79**: Tests that correction dose for BGL=90 is negative, but the implementation now has a dead-band — this test is likely broken (correction dose will be 0, not negative)

### Reproducibility Issues

1. `np.random.normal()` used directly in training (line 214 of agent, line 393 of train_a2c) instead of through a seeded RNG
2. Numba's `fastmath=True` allows non-deterministic floating-point reordering
3. TF eager mode has non-deterministic GPU/CPU kernel selection
4. No experiment tracking (no MLflow, no W&B, no sacred)
5. No config dumping — hyperparameters are logged to stdout but not saved

---

## 10. CRITICAL BUG LIST

### BUG-001: Numba/Python Integrator Inconsistency
- **Severity**: CRITICAL
- **Root cause**: Numba `hovorka_one_min_step` uses forward Euler without circadian EGP; Python path uses RK45 with circadian EGP
- **Affected files**: `src/ap_rl/envs/hovorka_patient.py:323-389` vs `216-272`
- **Fix**: Unify to a single integration method. Use RK4 (fixed-step) in Numba with circadian modulation included.

### BUG-002: PID PTerm Initialization
- **Severity**: HIGH
- **Root cause**: `PID.clear()` sets `self.PTerm = 0.2` instead of 0.0, biasing the first control output
- **Affected files**: `src/ap_rl/utils/pid_controller.py:39`
- **Fix**: Set `self.PTerm = 0.0`

### BUG-003: Reward-TIR Anti-Correlation
- **Severity**: CRITICAL
- **Root cause**: IOB brake and rate penalties dominate the reward, punishing good control achieved through aggressive (but effective) insulin delivery
- **Affected files**: `src/ap_rl/envs/diabetes_pid_env.py:368-476`
- **Fix**: Redesign reward to be primarily TIR-based with soft secondary penalties

### BUG-004: Correction Dose Test Failure
- **Severity**: MEDIUM
- **Root cause**: `test_insulin_calculator.py:79` expects negative correction dose for BGL=90, but dead-band (20 mg/dL) means glucose_difference (90-120=-30) is below target, so correction should be 0, not negative. The `calculate_correction_dose` returns 0 when `glucose_difference < correction_dead_band` (which equals 20). Since -30 < 20, it returns 0. But the test expects `dose < 0`.
- **Affected files**: `tests/test_insulin_calculator.py:79`
- **Fix**: Update test expectation to `== 0.0`

### BUG-005: Temp File Race Condition
- **Severity**: HIGH
- **Root cause**: `DiabetesPIDEnv.reset()` writes to fixed path `temp_data/meal_temp.data`. Parallel envs (multi-process training) would corrupt each other.
- **Affected files**: `src/ap_rl/envs/diabetes_pid_env.py:252-259`
- **Fix**: Pass meal data directly to HovorkaPatient via method or constructor, eliminating file I/O entirely.

### BUG-006: IOB Exponential Decay is Physiologically Wrong
- **Severity**: HIGH
- **Root cause**: `self.iob_units = (self.iob_units * 0.985) + (total_insulin_rate / 60.0)` models IOB as simple exponential decay with 45-min half-life. Real subcutaneous insulin has a characteristic absorption profile: onset 15min, peak 60-90min, duration 3-5h (rapid-acting). The exponential model drastically underestimates IOB during the first hour.
- **Affected files**: `src/ap_rl/envs/diabetes_pid_env.py:594`
- **Fix**: Implement a proper pharmacokinetic IOB model (e.g., Hovorka's own S1/S2 subcutaneous compartments provide this for free — just sum S1+S2+I).

### BUG-007: Kp Upper Bound Allows Dangerous Insulin Delivery
- **Severity**: HIGH
- **Root cause**: `Kp` clipped to [0.01, 10.0]. At Kp=10 with 40 mg/dL error, PID output is 400, scaled by 0.01 → 4 U/h basal adjustment. Combined with base_basal (~0.86 U/h for 75kg), total can reach ~5 U/h — dangerously high for many patients.
- **Affected files**: `src/ap_rl/envs/diabetes_pid_env.py:551`
- **Fix**: Reduce Kp upper bound to 3.0 or better, add a total insulin rate ceiling.

### BUG-008: `time.time()` Dependency in PID Constructor
- **Severity**: LOW
- **Root cause**: PID `__init__` uses `time.time()` for initial `current_time` and `last_time`. First `update()` call computes `delta_time = current_time - last_time` where `last_time` is a wall-clock timestamp but `current_time` is simulation minutes.
- **Affected files**: `src/ap_rl/utils/pid_controller.py:25-26`; `src/ap_rl/envs/diabetes_pid_env.py:573` (calls `self.pid.update(current_glucose)` without passing `current_time`)
- **Fix**: Always pass simulation time to PID; initialize PID with `current_time=0`

---

## 11. PRIORITY MATRIX

### MUST FIX (Blocks any meaningful results)

| Issue | Description |
|-------|-------------|
| Numba/Python integrator divergence | Produces different physics depending on Numba availability |
| Reward function redesign | Anti-correlated with clinical outcomes |
| RL formulation (PID-delta action space) | Fundamental architectural flaw |
| IOB model replacement | Current exponential decay is physiologically invalid |
| Temp file I/O elimination | Breaks parallelism, wastes I/O |
| PID time basis | Wall-clock dependency creates non-determinism |
| Safety constraints on total insulin | No upper bound on insulin delivery |

### SHOULD FIX (Significantly improves quality)

| Issue | Description |
|-------|-------------|
| Add clinical metrics (TBR, GMI, CV, LBGI/HBGI) | Current metrics miss critical safety information |
| PID PTerm initialization | Biases first control output |
| Kp range reduction | [0.01, 10] allows dangerous insulin delivery |
| Observation normalization | 7 of 19 features are near-zero |
| Separate env from controller | DiabetesPIDEnv God class |
| Add regression tests with golden trajectories | No way to detect silent changes |
| Fix Numba circadian EGP omission | Different day/night behavior |

### NICE TO HAVE (Polish and extensibility)

| Issue | Description |
|-------|-------------|
| Multi-day scenarios | Only 24-hour simulations currently |
| Sensor noise model (ARIMA/drift) | Current model is IID Gaussian |
| Meal composition effects | All carbs treated equally |
| Experiment tracking (W&B/MLflow) | Training runs are untracked |
| UI component architecture | Monolithic Streamlit script |
| WebSocket live playback | Current UI is static post-hoc |
| Controller introspection panels | No explainability |

### DELETE ENTIRELY

| Item | Reason |
|------|--------|
| `temp_data/` file I/O loop | Pure waste — data already in memory |
| `atexit` cleanup handler | Module-level side effect |
| `docs/legacy-pid-tuner/` | Lunar Lander code has no relation to AP |
| `TestCaseManager/` | Amesim-specific, violates self-contained requirement |
| `verify_best_model.py` | Duplicate of rollout.py with hardcoded paths |
| PID `time.time()` default | Wall-clock dependency in simulation code |
| `hovorka_gym_env.py` | Half-baked Gymnasium wrapper, incomplete |

---

*End of Audit Report*
