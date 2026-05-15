# FILE-BY-FILE REVIEW

## Quick Reference Table

| File | Quality | Verdict | Tech Debt | Key Issue |
|------|---------|---------|-----------|-----------|
| `envs/diabetes_pid_env.py` | 2/5 | **DELETE/REBUILD** | CRITICAL | God class, wrong RL formulation, temp file I/O |
| `envs/hovorka_patient.py` | 3/5 | REFACTOR | CRITICAL | Numba/Python integrator mismatch |
| `envs/hovorka_gym_env.py` | 1/5 | **DELETE** | HIGH | Half-baked, different obs space |
| `envs/defaults.py` | 4/5 | KEEP | LOW | Clean constants |
| `envs/profile_loader.py` | 4/5 | KEEP | LOW | Well-structured |
| `envs/scenario_builder.py` | 4/5 | KEEP | LOW | Clean |
| `agents/diabetes_a2c_actor.py` | 3/5 | REFACTOR | HIGH | Fixed std, wrong action space |
| `agents/diabetes_a2c_critic.py` | 3/5 | REFACTOR | MEDIUM | Port to PyTorch |
| `agents/diabetes_a2c_agent.py` | 2/5 | REFACTOR | HIGH | Double-update bug, monkey-patched |
| `training/train_a2c.py` | 2/5 | REFACTOR | HIGH | Monkey-patches agent.train() |
| `runtime/rollout.py` | 4/5 | KEEP | LOW | Clean EpisodeRecord |
| `utils/insulin_calculator.py` | 4/5 | KEEP | LOW | Sound clinical math |
| `utils/pid_controller.py` | 2/5 | REFACTOR | HIGH | PTerm=0.2, no filter, time.time() |
| `utils/meal_parser.py` | 3/5 | KEEP | MEDIUM | Fragile regex |
| `utils/config.py` | 5/5 | KEEP | LOW | Minimal, correct |
| `utils/paths.py` | 5/5 | KEEP | LOW | Well-designed fallback |
| `utils/seed.py` | 4/5 | KEEP | LOW | Clean |
| `utils/checkpoint_filenames.py` | 4/5 | KEEP | LOW | Keras3 compat |
| `utils/hardware.py` | 3/5 | DELETE (if PyTorch) | MEDIUM | TF-specific |
| `evaluation/metrics.py` | 4/5 | REFACTOR | MEDIUM | Missing clinical metrics |
| `visualization/publication.py` | 4/5 | KEEP | LOW | Well-parameterized |
| `app/app.py` | 2/5 | REFACTOR | HIGH | Monolithic, tight coupling |
| `demo.py` | 4/5 | KEEP | LOW | Simple launcher |
| `verify_best_model.py` | 1/5 | **DELETE** | HIGH | Duplicates rollout, hardcoded |
| Tests (all) | 3/5 | KEEP+EXPAND | MEDIUM | Smoke only, no integration |
| Configs (all YAML) | 4/5 | KEEP | LOW | Need more patient profiles |
| `docs/legacy-pid-tuner/` | N/A | **DELETE** | N/A | LunarLander — unrelated |
| `TestCaseManager/` | N/A | **DELETE** | N/A | Amesim-dependent |

---

## Detailed Reviews

### `src/ap_rl/envs/diabetes_pid_env.py`

**Purpose**: Main RL environment combining Hovorka patient simulation, PID control, insulin bolus management, reward computation, observation construction, scenario loading, and file I/O.

**Quality**: 2/5

**Major Problems**:
1. **God class** (674 lines, 8+ responsibilities): Should be decomposed into PatientSimulator, Controller, RewardFunction, ObservationBuilder, ScenarioLoader, InsulinManager.
2. **Wrong action space**: `(dKp, dKi, dKd)` deltas create 3 levels of indirection between RL action and insulin delivery.
3. **Reward anti-correlated with TIR**: IOB brake and rate penalties dominate, causing -93k reward at 96.7% TIR.
4. **Temp file I/O**: `reset()` writes to `temp_data/` then reads back — data already in memory.
5. **IOB model wrong**: Exponential decay (0.985^t) vs real pharmacokinetics.
6. **atexit handler**: Module-level side effect deletes temp_data/.
7. **Magic numbers**: 0.985, 0.01, 0.55, 18.0182, 400.0, 200.0 scattered without named constants.
8. **Duplicate assignments**: `bolus_duration` set to 30 then 15, `max_episode_length` set twice.
9. **Safety clamp ignores bolus insulin** — only constrains basal rate.
10. **Reward target (100) != PID target (120)** — creates internal tug-of-war.

**Verdict**: DELETE and rebuild as 5+ focused classes.

**Dependencies**: hovorka_patient, insulin_calculator, meal_parser, pid_controller, paths, metrics.

**Tech Debt**: CRITICAL

**Rewrite Strategy**: Decompose into:
- `simulation/simulator.py` — orchestrates patient model stepping
- `controllers/pid.py` — standalone PID with proper anti-windup
- `environments/glucose_control_env.py` — Gymnasium-compliant env with direct insulin action space
- `controllers/safety.py` — hard safety constraints
- `core/reward.py` — clean, bounded reward function

---

### `src/ap_rl/envs/hovorka_patient.py`

**Purpose**: 10-state Hovorka ODE model for Type-1 diabetes glucose-insulin dynamics.

**Quality**: 3/5

**Major Problems**:
1. **CRITICAL**: Numba forward Euler omits circadian EGP modulation (line 360 vs 250-253).
2. **CRITICAL**: Numba (Euler dt=1min) and Python (RK45 adaptive) produce different trajectories.
3. **t_max_G default mismatch**: Python ODE defaults to 40, Numba/config use 30.
4. Meal loading is instantaneous (`D1 += carbs_mmol`) — no gastric emptying model.
5. Exact-time meal matching (`meal_data['time'] == t`) vulnerable to float precision.
6. Exercise model only affects glucose uptake, not EGP or insulin sensitivity.

**Verdict**: REFACTOR — fix integration, unify paths, add gastric emptying.

**Dependencies**: numpy, scipy, pandas, numba (optional).

**Tech Debt**: CRITICAL

**Rewrite Strategy**: Use a single RK4 fixed-step integrator (4 sub-steps per minute) that works in both Numba and pure Python. Include circadian EGP in both. Add configurable meal absorption profile.

---

### `src/ap_rl/envs/hovorka_gym_env.py`

**Purpose**: Optional Gymnasium wrapper for the Hovorka patient model.

**Quality**: 1/5

**Major Problems**:
1. Different 13-D observation space vs main env's 19-D.
2. Not integrated with training pipeline.
3. Tests marked `importorskip` — treated as second-class.
4. Creates false impression of Gym compatibility.

**Verdict**: DELETE. Rebuild as the primary env interface in the new architecture.

---

### `src/ap_rl/envs/defaults.py`

**Purpose**: Default Hovorka parameter dictionary.

**Quality**: 4/5. Clean, well-commented constants.

**Verdict**: KEEP. Move to `core/constants.py` in new layout.

---

### `src/ap_rl/envs/profile_loader.py`

**Purpose**: Load patient profiles from YAML configs.

**Quality**: 4/5. Clean `PatientProfile` dataclass with sensible defaults.

**Verdict**: KEEP with minor refactor (move to `core/` or `simulation/`).

---

### `src/ap_rl/envs/scenario_builder.py`

**Purpose**: Build meal/exercise schedules from YAML scenario files.

**Quality**: 4/5. Simple, correct, well-structured.

**Verdict**: KEEP.

---

### `src/ap_rl/agents/diabetes_a2c_actor.py`

**Purpose**: TF/Keras actor network producing PID gain deltas.

**Quality**: 3/5

**Major Problems**:
1. Action space is PID deltas (wrong formulation).
2. Fixed `std = 0.1` for log-probability — policy entropy is constant.
3. Entropy bonus formula uses hardcoded constants that don't adapt.
4. `@tf.function` removed with comment about macOS freeze — framework stability concern.

**Verdict**: REFACTOR to PyTorch with learned log_std and direct insulin action space.

**Tech Debt**: HIGH

---

### `src/ap_rl/agents/diabetes_a2c_critic.py`

**Purpose**: TF/Keras state-value network.

**Quality**: 3/5. Simple and functional, but TF-dependent.

**Verdict**: REFACTOR to PyTorch. Add LayerNorm for consistency with actor.

**Tech Debt**: MEDIUM

---

### `src/ap_rl/agents/diabetes_a2c_agent.py`

**Purpose**: A2C agent with training loop, evaluation, checkpointing.

**Quality**: 2/5

**Major Problems**:
1. `train()` is 160 lines mixing training, logging, checkpointing, evaluation.
2. `best_avg_reward` updated in two separate code paths — logic is confusing.
3. Early stopping patience counter interactions are error-prone.
4. Monkey-patched by `allprofiles` preset.
5. `np.random.normal` used directly (unseeded).

**Verdict**: REFACTOR. Separate training loop, checkpointing, and evaluation into distinct modules.

**Tech Debt**: HIGH

---

### `src/ap_rl/training/train_a2c.py`

**Purpose**: Training entry point with named presets and multi-profile curriculum.

**Quality**: 2/5

**Major Problems**:
1. Monkey-patches `agent.train` for allprofiles (duplicates 150 lines of training logic).
2. `eval_env_factory` is a lambda closing over mutable state.
3. `np.random.normal` used directly instead of seeded RNG.
4. No experiment tracking — hyperparams logged to stdout only.

**Verdict**: REFACTOR. Use a configurable Trainer class with patient sampling callback.

**Tech Debt**: HIGH

---

### `src/ap_rl/runtime/rollout.py`

**Purpose**: Episode runner producing `EpisodeRecord` dataclass.

**Quality**: 4/5. Clean design, proper auto-detection of state dim from checkpoint.

**Minor Issues**: Imports `DiabetesActor` at top level (forces TF import). State truncation for old models is a good backward-compat feature.

**Verdict**: KEEP with lazy TF import.

**Tech Debt**: LOW

---

### `src/ap_rl/utils/insulin_calculator.py`

**Purpose**: Clinical insulin dosing math (500/1500 rules, ISF-adaptive split bolus).

**Quality**: 4/5. Well-documented, sound clinical heuristics, proper lockout and tail dosing.

**Verdict**: KEEP. This is one of the strongest modules.

**Tech Debt**: LOW

---

### `src/ap_rl/utils/pid_controller.py`

**Purpose**: Vendored IvPID controller.

**Quality**: 2/5

**Major Problems**:
1. `PTerm` initialized to 0.2 in `clear()` — should be 0.0.
2. No derivative filtering — noise amplification.
3. `time.time()` default — wall-clock dependency.
4. Symmetric windup guard — should be asymmetric for insulin.
5. `sample_time` set but never enforced.

**Verdict**: REFACTOR or replace with a proper discrete-time PID.

**Tech Debt**: HIGH

---

### `src/ap_rl/utils/meal_parser.py`

**Purpose**: Parse TestCases.txt verbose format.

**Quality**: 3/5. Works but uses fragile regex/string splitting.

**Verdict**: KEEP for legacy compatibility. Add structured scenario format.

**Tech Debt**: MEDIUM

---

### `src/ap_rl/utils/config.py`

**Purpose**: YAML loader + deep merge.

**Quality**: 5/5. Minimal, correct, no unnecessary dependencies.

**Verdict**: KEEP as-is.

---

### `src/ap_rl/utils/paths.py`

**Purpose**: Repo root and data directory resolution.

**Quality**: 5/5. Well-thought-out fallback chain with env var override.

**Verdict**: KEEP as-is.

---

### `src/ap_rl/utils/seed.py`

**Purpose**: Global RNG seeding.

**Quality**: 4/5. Covers Python, NumPy, TF lazily.

**Verdict**: KEEP. Add PyTorch seeding when migrating.

---

### `src/ap_rl/utils/checkpoint_filenames.py`

**Purpose**: Keras 3 `.weights.h5` filename management.

**Quality**: 4/5. Clean backward compatibility logic.

**Verdict**: KEEP if staying with Keras, DELETE if migrating to PyTorch (use `.pt`).

---

### `src/ap_rl/utils/hardware.py`

**Purpose**: TensorFlow CPU threading configuration for M4 Mac.

**Quality**: 3/5. Well-commented but TF-specific.

**Verdict**: DELETE if migrating to PyTorch. Replace with PyTorch MPS/CPU config.

---

### `src/ap_rl/evaluation/metrics.py`

**Purpose**: Glucose trajectory metrics (TIR, mean, std, band percentages).

**Quality**: 4/5. Correct and clean, but incomplete.

**Missing**: TBR Level 2 (<54), GMI, CV%, LBGI, HBGI, MAGE, AGP statistics.

**Verdict**: REFACTOR to add full International Consensus 2019 metrics.

---

### `src/ap_rl/visualization/publication.py`

**Purpose**: matplotlib publication-quality plotting helpers.

**Quality**: 4/5. Clean API, proper rcParams, save_path support.

**Verdict**: KEEP. Add AGP report generation.

---

### `app/app.py`

**Purpose**: Streamlit digital twin demo.

**Quality**: 2/5

**Major Problems**:
1. Monolithic 450+ lines — all plotting, state, env construction in one file.
2. Monkey-patches env (`_skip_reload = True`).
3. Accesses env internals (`env.patient.time`).
4. No error boundaries — simulation crash = app crash.
5. Misleading "Digital Twin" label.
6. `@st.cache_resource` for TF model is fragile.

**Verdict**: REFACTOR into FastAPI backend + thin Streamlit client.

**Tech Debt**: HIGH

---

### `verify_best_model.py`

**Purpose**: Ad-hoc model verification script.

**Quality**: 1/5. Hardcoded paths, `sys.path.append("src")`, duplicates rollout.py.

**Verdict**: DELETE. Replace with proper evaluation CLI.

---

### `docs/legacy-pid-tuner/`

**Purpose**: Original LunarLander A2C/PPO experiments.

**Verdict**: DELETE entirely — unrelated to AP.

---

### `TestCaseManager/`

**Purpose**: Amesim test case generation scripts.

**Verdict**: DELETE entirely — violates self-contained requirement.

---

### Tests (`tests/*.py`)

**Purpose**: Smoke tests for env, metrics, insulin calculator, plots.

**Quality**: 3/5. Decent coverage of individual components.

**Missing**:
- Integration tests (full training loop)
- Regression tests (golden trajectories)
- Property-based tests (hypothesis)
- Performance benchmarks
- BUG-004: `test_correction_dose_negative` likely broken by dead-band.

**Verdict**: KEEP and significantly expand.

---

### Configs (`configs/*.yaml`)

**Purpose**: Patient profiles, meal templates, demo defaults.

**Quality**: 4/5. Clean, well-structured YAML.

**Limitation**: Only 4 patient profiles. Need 10-20 for realistic population coverage.

**Verdict**: KEEP and expand with more profiles and scenarios.

---

*End of File Review*
