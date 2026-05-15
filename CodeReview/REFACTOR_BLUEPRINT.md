# COMPLETE REFACTOR BLUEPRINT

## Artificial Pancreas RL-Tuned Controller — Redesign Specification

---

## 1. RECOMMENDED NEW ARCHITECTURE

### Core Design Philosophy

**Abandon the PID-delta RL formulation.** Replace with a **hierarchical hybrid controller**:

```
Layer 3: RL Supervisory Agent (slow: every 15-30 min)
  → Sets insulin delivery mode, aggressiveness level, meal override parameters
  
Layer 2: Model Predictive Controller (medium: every 5 min)
  → Computes optimal basal rate using glucose forecast + IOB prediction
  → Respects constraints from Layer 3
  
Layer 1: Safety Supervisor (fast: every 1 min)
  → Hard insulin constraints, hypo prediction, emergency suspend
  → Cannot be overridden by any higher layer
```

**Why this architecture:**

1. **MPC handles the core control problem** — it naturally handles insulin delay via its prediction horizon, respects constraints, and produces directly interpretable insulin commands.

2. **RL adds intelligence where MPC is weak** — meal detection, patient adaptation, inter-day sensitivity learning. RL operates at a slower timescale (15-30 min decisions), which makes the MDP far more tractable.

3. **Safety supervisor is hardware-guaranteed** — implemented as a separate, auditable module with formal correctness properties. No learned component can override it.

This is the architecture used by real commercial AP systems (Medtronic 780G, Tandem Control-IQ, Omnipod 5) and the academic state of the art.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT MANAGER                        │
│  (configs, seeds, logging, W&B/MLflow, scenario selection)  │
└─────────────┬───────────────────────────────┬───────────────┘
              │                               │
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────────┐
│   TRAINING ENGINE    │        │    EVALUATION ENGINE      │
│  (PPO/SAC + curriculum)│      │  (Monte Carlo, stress     │
│  (imitation pretraining)│     │   tests, clinical metrics) │
└──────────┬───────────┘        └──────────┬───────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   CONTROLLER ABSTRACTION                      │
│  ┌─────────┐  ┌────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ PID     │  │ MPC    │  │ Hybrid       │  │ Oracle     │ │
│  │ (base-  │  │ (pred- │  │ MPC+RL       │  │ (perfect   │ │
│  │  line)  │  │  ictive)│  │ (recommended)│  │  foresight)│ │
│  └─────────┘  └────────┘  └──────────────┘  └────────────┘ │
│                       │                                      │
│              ┌────────▼────────┐                            │
│              │ SAFETY SUPERVISOR│  ← hard constraints       │
│              │ (always active)  │  ← cannot be bypassed     │
│              └────────┬────────┘                            │
└───────────────────────┼──────────────────────────────────────┘
                        │ insulin command (U/h)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                    SIMULATION CORE                            │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │ Patient Model │  │ Insulin PK │  │ Sensor/Pump Models   │ │
│  │ (Hovorka ODE) │  │ (2-comp    │  │ (noise, delay, drift)│ │
│  │ (Bergman)     │  │  model)    │  │                      │ │
│  │ (UVa/Padova)  │  │            │  │                      │ │
│  └──────────────┘  └────────────┘  └──────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Scenario Engine (meals, exercise, stress, sensor faults) ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  VISUALIZATION SERVER                         │
│  (FastAPI backend + React frontend OR Streamlit-lite)        │
│  Live charts, replay, controller introspection, comparison   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. PROPOSED FOLDER STRUCTURE

```
ap-rl/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── configs/
│   ├── default.yaml              # master config (all defaults)
│   ├── patients/
│   │   ├── hovorka_default.yaml
│   │   ├── insulin_resistant.yaml
│   │   ├── insulin_sensitive.yaml
│   │   ├── unstable.yaml
│   │   └── pediatric.yaml
│   ├── scenarios/
│   │   ├── normal_day.yaml
│   │   ├── heavy_meals.yaml
│   │   ├── exercise_day.yaml
│   │   ├── sick_day.yaml
│   │   ├── overnight.yaml
│   │   └── stress_test.yaml
│   ├── controllers/
│   │   ├── pid_baseline.yaml
│   │   ├── mpc_default.yaml
│   │   ├── hybrid_mpc_rl.yaml
│   │   └── rl_only.yaml
│   └── training/
│       ├── ppo_default.yaml
│       ├── sac_default.yaml
│       ├── imitation_pretrain.yaml
│       └── curriculum.yaml
│
├── src/
│   └── ap_rl/
│       ├── __init__.py
│       ├── core/                   # ← NEW: shared abstractions
│       │   ├── __init__.py
│       │   ├── types.py            # dataclasses: PatientState, InsulinCommand, GlucoseReading
│       │   ├── constants.py        # named physiological constants
│       │   ├── units.py            # unit conversion helpers (mmol↔mg/dL, U/h↔U/min)
│       │   └── config.py           # typed config loading with pydantic/dataclasses
│       │
│       ├── simulation/             # ← REPLACES envs/
│       │   ├── __init__.py
│       │   ├── patient.py          # PatientModel ABC
│       │   ├── hovorka.py          # Hovorka ODE (single integrator, no Numba/Python split)
│       │   ├── bergman.py          # Optional: Bergman minimal model
│       │   ├── integrators.py      # RK4, RK45 fixed/adaptive steppers
│       │   ├── insulin_pk.py       # 2-compartment insulin pharmacokinetics
│       │   ├── sensor.py           # CGM sensor model (noise, delay, drift, calibration)
│       │   ├── pump.py             # Insulin pump model (min delivery, quantization)
│       │   ├── meal_model.py       # Gastric emptying + absorption (variable rate)
│       │   ├── exercise.py         # Exercise effect model
│       │   └── scenario.py         # Scenario builder: meals, exercise, faults
│       │
│       ├── controllers/            # ← NEW: pluggable controller system
│       │   ├── __init__.py
│       │   ├── base.py             # Controller ABC: observe(state) → InsulinCommand
│       │   ├── pid.py              # Standard PID with proper anti-windup
│       │   ├── mpc.py              # Model Predictive Controller (cvxpy or casadi)
│       │   ├── rl_agent.py         # RL policy wrapper (framework-agnostic)
│       │   ├── hybrid.py           # MPC + RL supervisory hybrid
│       │   ├── oracle.py           # Perfect-information controller (upper bound)
│       │   └── safety.py           # Safety supervisor (hard constraints, hypo prediction)
│       │
│       ├── environments/           # ← Gymnasium-compatible env wrappers
│       │   ├── __init__.py
│       │   ├── glucose_control_env.py   # Direct insulin → glucose (recommended for RL)
│       │   ├── supervisory_env.py       # RL supervisor over MPC (recommended hybrid)
│       │   └── wrappers.py              # Normalization, frame-stacking, etc.
│       │
│       ├── agents/                 # ← RL agent implementations
│       │   ├── __init__.py
│       │   ├── ppo.py              # PPO with clipped objective
│       │   ├── sac.py              # SAC for continuous control
│       │   ├── networks.py         # Shared actor/critic architectures
│       │   └── replay.py           # Experience replay buffers
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # Generic training loop (agent-agnostic)
│       │   ├── curriculum.py       # Curriculum learning schedule
│       │   ├── imitation.py        # BC/DAgger from expert MPC
│       │   └── callbacks.py        # Checkpointing, early stopping, W&B logging
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py          # Clinical metrics (TIR, TBR, GMI, CV, LBGI, HBGI, MAGE)
│       │   ├── monte_carlo.py      # Statistical testing across N scenarios
│       │   ├── stress_tests.py     # Adversarial scenario generation
│       │   └── comparison.py       # Controller A vs B statistical comparison
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── glucose_plot.py     # Glucose trajectory with bands
│       │   ├── insulin_plot.py     # Insulin delivery decomposition
│       │   ├── agp_report.py       # Ambulatory Glucose Profile (clinical standard)
│       │   ├── controller_viz.py   # Controller state introspection
│       │   └── comparison.py       # Multi-controller overlay
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── paths.py
│       │   ├── seed.py
│       │   └── hardware.py
│       │
│       └── app/                    # ← Demo application
│           ├── __init__.py
│           ├── server.py           # FastAPI backend (simulation API)
│           ├── streamlit_app.py    # Streamlit frontend (thin client)
│           └── components/         # Reusable UI components
│               ├── glucose_chart.py
│               ├── insulin_chart.py
│               ├── scenario_editor.py
│               ├── controller_panel.py
│               └── metrics_dashboard.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_hovorka.py         # ODE correctness vs reference
│   │   ├── test_insulin_pk.py
│   │   ├── test_pid.py
│   │   ├── test_mpc.py
│   │   ├── test_safety.py
│   │   ├── test_metrics.py
│   │   └── test_scenario.py
│   ├── integration/
│   │   ├── test_closed_loop.py     # Full controller + patient loop
│   │   ├── test_training_smoke.py  # 2-episode training convergence
│   │   └── test_reproducibility.py # Seeded runs match exactly
│   ├── regression/
│   │   └── golden_trajectories/    # Known-good outputs for comparison
│   └── stress/
│       └── test_edge_cases.py      # Extreme meals, zero insulin, sensor failure
│
├── scripts/
│   ├── train.py                    # CLI training entry point
│   ├── evaluate.py                 # CLI evaluation entry point
│   ├── demo.py                     # Launch demo server
│   └── generate_scenarios.py       # Bulk scenario generation
│
├── notebooks/
│   ├── 01_patient_model_validation.ipynb
│   ├── 02_controller_comparison.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_publication_figures.ipynb
│
└── docs/
    ├── architecture.md
    ├── controllers.md
    ├── patient_models.md
    └── api_reference.md
```

---

## 3. PROPOSED TECH STACK

| Component | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **RL Framework** | Raw TF/Keras | **Stable-Baselines3 (PyTorch)** or **CleanRL** | SB3 has validated PPO/SAC implementations. PyTorch is more debuggable than TF. CleanRL for single-file simplicity. |
| **Deep Learning** | TensorFlow 2.x | **PyTorch 2.x** | Better debugging, JIT compilation, M-series Metal support via `torch.mps` |
| **MPC Solver** | None | **CasADi** or **cvxpy** | CasADi: fast nonlinear MPC with automatic differentiation. cvxpy: simpler for linear/QP formulations. |
| **ODE Integration** | scipy + Numba Euler | **torchdiffeq** or **diffrax (JAX)** | Differentiable ODE solvers enable end-to-end gradient-based optimization through the patient model |
| **Configuration** | Plain YAML | **Hydra** or **pydantic + YAML** | Hydra: hierarchical configs with CLI overrides. pydantic: typed validation. |
| **Experiment Tracking** | None (stdout) | **Weights & Biases** or **MLflow** | W&B: best UX, free for academics. MLflow: self-hosted, no vendor lock-in. |
| **Plotting** | matplotlib + Plotly | **Plotly** (interactive) + **matplotlib** (publication) | Keep both — Plotly for demo, matplotlib for papers |
| **Web Backend** | Streamlit (monolithic) | **FastAPI** + **Streamlit** (thin client) | FastAPI serves simulation API; Streamlit or React consumes it. Decouples compute from presentation. |
| **Testing** | pytest (smoke only) | **pytest** + **hypothesis** + **pytest-benchmark** | hypothesis for property-based testing, benchmark for performance regression |
| **Type Checking** | None | **mypy** (strict) | Catch unit mismatches, None errors, shape mismatches at lint time |
| **Gymnasium** | Optional, half-baked | **Gymnasium** (required, standard) | All envs implement Gymnasium API. Enables SB3 integration. |

---

## 4. CONTROLLER STRATEGY RECOMMENDATION

### Should RL Remain the Primary Controller?

**No.** RL should be **demoted from primary controller to supervisory intelligence layer.**

**Evidence:**
1. The PID-delta formulation is mathematically ill-conditioned (Section 3 of audit)
2. After 600 training episodes, TIR is 78-96% — a well-tuned MPC achieves >90% TIR with zero training
3. The 45-minute insulin delay makes the MDP partially observable with extremely delayed rewards
4. RL adds unpredictable behaviors that are unacceptable in safety-critical systems

### Recommended Controller Hierarchy

#### Tier 1: Model Predictive Control (Primary)

MPC is the correct tool for insulin dosing because:
- It explicitly handles the insulin absorption delay via its prediction horizon
- It respects hard constraints (min/max insulin, IOB limits) naturally
- It uses the patient model (Hovorka) as its internal predictor
- It produces interpretable, optimal control actions
- It works immediately without training

**MPC Specification:**
```
Prediction horizon: 180 minutes (3 hours)
Control horizon: 30 minutes
Sampling: 5-minute intervals
Decision variable: basal rate u(t) ∈ [0, max_basal] U/h
Cost function: 
  J = Σ[ w_glucose * (G(t) - G_target)² 
      + w_hypo * max(0, G_low - G(t))²    # asymmetric hypo penalty
      + w_insulin * Δu(t)²                  # smooth insulin changes
      + w_iob * max(0, IOB(t) - IOB_max)²  # IOB constraint
      ]
Subject to:
  G(t+1) = f(G(t), u(t))  # Hovorka model
  0 ≤ u(t) ≤ u_max
  IOB(t) ≤ IOB_max
```

#### Tier 2: RL Supervisory Agent

The RL agent operates at a **slower timescale** (every 15-30 minutes) and adjusts:
- MPC cost function weights (aggressiveness)
- Meal detection confidence → trigger bolus recommendation
- Patient sensitivity estimate → adapt MPC model parameters
- Target glucose setpoint (within safe range)

**RL Specification:**
```
Algorithm: PPO (stable, well-tested for continuous control) or SAC
State: [glucose_30min_history, IOB, time_of_day, meal_detected, 
        current_mpc_weights, patient_sensitivity_estimate]
Action: [target_glucose_adjustment, aggressiveness_level, 
         sensitivity_multiplier]  ∈ continuous, bounded
Reward: TIR-based (70-180) with small hypo penalty
Episode: 24 hours (288 steps at 5-min intervals)
```

This formulation is **orders of magnitude easier** for RL because:
- 288 steps/episode vs 1440 (4× fewer decisions)
- Actions directly meaningful (not gain deltas)
- Reward well-correlated with clinical outcomes
- MPC handles the hard control problem; RL only needs to adapt

#### Tier 3: Safety Supervisor (Always Active)

```python
class SafetySupervisor:
    """Hard safety constraints — no learned component can override."""
    
    MAX_BASAL_UH = 5.0          # U/h absolute maximum
    MAX_BOLUS_U = 15.0          # U single bolus maximum
    MAX_IOB_U = 20.0            # U maximum insulin on board
    SUSPEND_GLUCOSE = 70.0      # mg/dL: full insulin suspension
    REDUCE_GLUCOSE = 90.0       # mg/dL: reduce to 50% basal
    PREDICTIVE_HORIZON = 30     # minutes ahead for predictive suspend
    PREDICTIVE_THRESHOLD = 80.0 # mg/dL: suspend if predicted to fall below
    
    def constrain(self, command: InsulinCommand, state: PatientState) -> InsulinCommand:
        # 1. Hard clamp on total insulin
        command.basal = min(command.basal, self.MAX_BASAL_UH)
        command.bolus = min(command.bolus, self.MAX_BOLUS_U)
        
        # 2. IOB constraint
        if state.iob >= self.MAX_IOB_U:
            command.basal = 0.0
            command.bolus = 0.0
            
        # 3. Reactive suspension
        if state.glucose < self.SUSPEND_GLUCOSE:
            command.basal = 0.0
            
        # 4. Predictive suspension (linear extrapolation)
        predicted = state.glucose + state.glucose_rate * self.PREDICTIVE_HORIZON
        if predicted < self.PREDICTIVE_THRESHOLD:
            command.basal = 0.0
            
        # 5. Reduction zone
        if state.glucose < self.REDUCE_GLUCOSE:
            command.basal *= 0.5
            
        return command
```

### Whether Ensemble Methods Help

**Yes, for uncertainty estimation.** An ensemble of 3-5 MPC models with slightly different patient parameters provides:
- Uncertainty bounds on glucose predictions
- Robust control (worst-case optimization across ensemble)
- Online patient identification (which ensemble member best matches observations)

### Whether Safety Supervisors Are Required

**Absolutely required and non-negotiable.** Any system that controls insulin delivery must have a hard-coded safety layer that cannot be overridden by learned components. This is not optional.

---

## 5. SAFETY LAYER DESIGN

### Insulin Hard Constraints

```python
CONSTRAINTS = {
    "max_basal_uh": 5.0,               # absolute ceiling
    "max_bolus_u": 15.0,               # single bolus max
    "max_iob_u": 20.0,                 # total insulin on board
    "min_bolus_interval_min": 15,       # lockout between boluses
    "max_daily_insulin_u": 100.0,       # 24-hour cumulative limit
    "max_insulin_rate_change_uh_min": 2.0,  # rate of change limit
}
```

### Hypo Prevention (Multi-Layer)

1. **Reactive Layer** (always active):
   - Glucose < 54 mg/dL: full suspension + alert
   - Glucose < 70 mg/dL: full suspension
   - Glucose < 80 mg/dL AND falling: full suspension

2. **Predictive Layer** (30-minute lookahead):
   - Linear extrapolation from last 15 minutes
   - If predicted glucose < 80 mg/dL: reduce to 30% basal
   - If predicted glucose < 70 mg/dL: full suspension

3. **IOB-Aware Layer**:
   - Compute expected glucose drop from current IOB: `expected_drop = IOB * ISF`
   - If `current_glucose - expected_drop < 80`: reduce insulin proportionally

### Risk-Aware Penalties (for RL reward)

Replace the current chaotic reward with a clean clinical objective:

```python
def clinical_reward(glucose: float, iob: float, insulin_rate: float) -> float:
    """Reward based on International Consensus Guidelines."""
    # Primary: Time in range (smooth approximation)
    if 70 <= glucose <= 180:
        reward = 1.0  # full credit
    elif 54 <= glucose < 70:
        reward = -2.0 * (70 - glucose) / 16   # linear, max -2
    elif glucose < 54:
        reward = -5.0 - (54 - glucose) * 0.5  # severe, max ~-30
    elif 180 < glucose <= 250:
        reward = -0.5 * (glucose - 180) / 70   # linear, max -0.5
    else:
        reward = -1.0 - (glucose - 250) * 0.02 # moderate
    
    # Secondary: insulin smoothness (prevent oscillation)
    reward -= 0.01 * abs(insulin_rate_change)
    
    return reward
```

Key design principles:
- Rewards bounded to approximately [-30, +1] per step
- Hypo penalty ~10× hyper penalty (matching clinical importance)
- No catastrophic -10,000 spikes
- No complex interaction terms

### Fallback Controllers

If the RL supervisor or MPC fails (NaN, crash, timeout):
```
Fallback chain: MPC → PID (safe gains) → Basal-only → Suspend
```

Each level is simpler and safer. The system always has a working controller.

---

## 6. DIGITAL TWIN DESIGN

### Patient Variability

**Population model** using log-normal distributions for Hovorka parameters:

```python
POPULATION_PARAMS = {
    # parameter: (mean, cv%)  — CV from published Hovorka population data
    "BW":     (75.0, 20),     # 55-100 kg
    "k_e":    (0.138, 25),    # insulin elimination
    "EGP_0":  (0.0161, 20),   # endogenous glucose production
    "V_G":    (0.16, 15),     # glucose distribution volume
    "k_12":   (0.066, 30),    # glucose transfer rate
    "F_01":   (0.0097, 20),   # non-insulin-dependent glucose flux
    "t_max_I": (55, 25),      # insulin absorption time constant
    "t_max_G": (40, 30),      # meal absorption time constant
}

def sample_patient(rng: np.random.Generator) -> dict:
    """Sample a physiologically plausible virtual patient."""
    params = {}
    for key, (mean, cv_pct) in POPULATION_PARAMS.items():
        sigma = mean * cv_pct / 100
        params[key] = max(mean * 0.3, rng.lognormal(np.log(mean), sigma/mean))
    return params
```

### Domain Randomization (During Training)

Each training episode randomly varies:
- Patient parameters (from population model)
- Meal timing (±30 min jitter)
- Carb amounts (±30% counting error)
- Insulin sensitivity (0.7× to 1.4× daily variation)
- Sensor noise characteristics (std 5-15 mg/dL)
- Sensor delay (5-15 minutes)

### Stochastic Meals

```yaml
# Meal template with uncertainty
breakfast:
  time_mean: 480        # 8:00 AM
  time_std: 30          # ±30 min
  carbs_mean: 45
  carbs_std: 15         # ±33% counting error
  glycemic_index: 0.7   # affects absorption rate
  fat_content: 0.3      # slows gastric emptying
```

### Sensor Noise Model

Replace IID Gaussian with realistic CGM model:
```python
class CGMSensor:
    """Dexcom-like CGM sensor model."""
    def __init__(self, mard_pct=9.0, delay_min=10, drift_rate=0.5):
        self.mard = mard_pct / 100  # Mean Absolute Relative Difference
        self.delay_buffer = deque(maxlen=delay_min)
        self.drift = 0.0
        self.drift_rate = drift_rate  # mg/dL per hour
    
    def read(self, true_glucose: float, rng) -> float:
        self.delay_buffer.append(true_glucose)
        delayed = self.delay_buffer[0]
        noise = rng.normal(0, delayed * self.mard)
        self.drift += rng.normal(0, self.drift_rate / 60)
        return max(40, delayed + noise + self.drift)
```

---

## 7. TRAINING PIPELINE REDESIGN

### Phase 1: Imitation Learning from Expert MPC (Days 1-2)

Train the RL policy to imitate a well-tuned MPC controller:
```python
# Generate expert demonstrations
expert_mpc = MPCController(patient_model=hovorka, horizon=180)
demonstrations = []
for scenario in all_scenarios:
    trajectory = expert_mpc.run(scenario)
    demonstrations.append(trajectory)

# Behavioral cloning
policy.pretrain(demonstrations, epochs=100)  # warm-start the policy
```

**Why**: RL from scratch in this domain is inefficient. Starting from an MPC-imitating policy ensures the agent already knows reasonable insulin dosing. RL then fine-tunes for scenarios where MPC is suboptimal.

### Phase 2: Curriculum RL Training (Days 3-10)

```python
curriculum = [
    # Stage 1: Easy (nominal patient, simple meals)
    {"episodes": 200, "patients": "nominal", "meals": "3_regular"},
    # Stage 2: Medium (varied patients, varied meals)
    {"episodes": 300, "patients": "population_sample", "meals": "random_3_5"},
    # Stage 3: Hard (extreme patients, stress scenarios)
    {"episodes": 500, "patients": "extreme", "meals": "adversarial"},
]
```

### Phase 3: Robust Fine-Tuning (Days 11-15)

- Adversarial training: worst-case meal timing/amounts
- Noise injection: sensor failures, insulin delivery errors
- Cross-validation: train on 80% of patient population, validate on 20%

### Replay Strategy

Use **Prioritized Experience Replay** with priority proportional to:
- TD error magnitude (standard PER)
- Clinical severity: episodes containing hypo events have 5× priority
- Scenario novelty: rare patient types / meal patterns prioritized

### Reward Redesign

```python
def reward_v2(glucose, target=110, iob=0, insulin_change=0):
    """Clean, bounded, clinically-aligned reward."""
    # TIR component: +1 in range, negative outside
    if 70 <= glucose <= 180:
        zone_reward = 1.0 - 0.5 * ((glucose - target) / 70) ** 2
    elif glucose < 70:
        zone_reward = -3.0 * ((70 - glucose) / 30) ** 2  # -3 at glucose=40
    else:
        zone_reward = -0.5 * ((glucose - 180) / 100) ** 2  # -0.5 at glucose=280
    
    # Smoothness penalty (small)
    smoothness = -0.05 * abs(insulin_change)
    
    return zone_reward + smoothness  # bounded to approximately [-3, +1]
```

---

## 8. EVALUATION FRAMEWORK

### Monte Carlo Testing

```python
def monte_carlo_evaluation(controller, n_patients=100, n_scenarios=50, seed=42):
    """Run N_patients × N_scenarios simulations and compute population statistics."""
    results = []
    for patient in sample_population(n_patients, seed):
        for scenario in sample_scenarios(n_scenarios, seed):
            trajectory = simulate(controller, patient, scenario)
            metrics = compute_clinical_metrics(trajectory)
            results.append(metrics)
    
    return {
        "population_tir": np.mean([r["tir_70_180"] for r in results]),
        "population_tbr_54": np.mean([r["tbr_54"] for r in results]),
        "worst_case_tir": np.min([r["tir_70_180"] for r in results]),
        "worst_case_hypo_minutes": np.max([r["minutes_below_54"] for r in results]),
        # ... full international consensus metrics
    }
```

### Clinical Metrics (International Consensus 2019)

| Metric | Target | Implementation |
|--------|--------|----------------|
| TIR (70-180 mg/dL) | >70% | `percent_in_range(glucose, 70, 180)` |
| TBR Level 1 (54-70) | <4% | `percent_in_range(glucose, 54, 70)` |
| TBR Level 2 (<54) | <1% | `percent_below(glucose, 54)` |
| TAR Level 1 (180-250) | <25% | `percent_in_range(glucose, 180, 250)` |
| TAR Level 2 (>250) | <5% | `percent_above(glucose, 250)` |
| CV | <36% | `np.std(glucose) / np.mean(glucose) * 100` |
| GMI | <7.0% | `3.31 + 0.02392 * mean_glucose_mgdl` |
| LBGI / HBGI | — | Kovatchev risk functions |

### Stress Tests

```python
stress_scenarios = [
    "massive_meal_100g",       # 100g carbs, no bolus
    "double_bolus",            # accidental double bolus delivery
    "sensor_dropout_60min",    # 60-min sensor gap
    "exercise_unannounced",    # 45-min vigorous exercise, no warning
    "dawn_phenomenon_severe",  # 4 AM glucose rise +80 mg/dL
    "insulin_site_failure",    # 50% absorption reduction
    "carb_miscount_2x",       # reported 30g, actual 60g
    "overnight_hypo_risk",    # late-night bolus, no snack
]
```

---

## 9. UI REDESIGN

### Architecture: FastAPI Backend + Streamlit Thin Client

```python
# server.py — FastAPI simulation API
@app.post("/simulate")
async def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a complete simulation and return results."""
    patient = create_patient(config.patient)
    controller = create_controller(config.controller)
    trajectory = run_simulation(patient, controller, config.scenario)
    metrics = compute_metrics(trajectory)
    return SimulationResult(trajectory=trajectory, metrics=metrics)

@app.post("/simulate/stream")
async def simulate_stream(config: SimulationConfig):
    """Stream simulation results minute-by-minute via SSE."""
    async for step in run_simulation_async(config):
        yield ServerSentEvent(data=step.json())
```

### UI Flow

1. **Scenario Setup** (sidebar):
   - Patient profile selector (with parameter preview)
   - Meal schedule editor (drag-and-drop timeline)
   - Exercise schedule editor
   - Controller selector (PID / MPC / Hybrid / RL)

2. **Simulation Control** (top bar):
   - Play / Pause / Step / Reset buttons
   - Speed slider (1× to 100× real-time)
   - Time scrubber

3. **Main Dashboard** (center):
   - **Glucose chart**: real-time trace with confidence bands, meal markers, exercise spans
   - **Insulin chart**: basal + bolus decomposition
   - **Controller state**: PID gains, MPC prediction horizon, RL action distribution

4. **Metrics Panel** (right):
   - Live TIR gauge (updating as simulation progresses)
   - Clinical metrics table (TIR, TBR, TAR, CV, GMI)
   - Distribution histogram (hypo/in-range/hyper)

5. **Comparison Mode**:
   - Run same scenario with multiple controllers simultaneously
   - Side-by-side or overlay view
   - Statistical comparison table

6. **Explainability Panel** (expandable):
   - MPC predicted glucose trajectory (180-min forecast)
   - RL action probability distribution
   - Safety supervisor state (active constraints)
   - IOB compartment visualization
   - Insulin absorption curve

---

*End of Refactor Blueprint*
