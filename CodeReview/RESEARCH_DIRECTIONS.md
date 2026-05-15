# Research Directions — Artificial Pancreas RL Controller

> **Project context.** This repository implements an A2C reinforcement learning agent
> that tunes PID controller gains (Kp, Ki, Kd) for automated insulin delivery in a
> Type 1 Diabetes simulation built on the Hovorka virtual patient ODE model. The
> system is being redesigned toward a hierarchical hybrid MPC+RL architecture where
> MPC handles core glucose control, RL provides supervisory adaptation, and a
> hard-coded safety layer enforces clinical constraints. This document maps out
> research directions that extend the project toward publication-quality results and
> a compelling portfolio demonstration.
>
> **Current performance.** Held-out mean TIR (70--180 mg/dL) of 84.7% +/- 7.9%
> across 12 unseen patient scenarios; rage-bolus crashes eliminated; IOB-aware
> reward shaping operational. The existing A2C agent outputs incremental
> (delta-Kp, delta-Ki, delta-Kd) every 5 minutes in a 19-dimensional observation
> space with clinical zone rewards.
>
> **Not a medical device.** All work described here is research/engineering
> demonstration only.

---

## Priority Recommendations

**If pursuing 3--4 directions for maximum combined impact, start here:**

| Priority | Direction | Rationale |
|----------|-----------|-----------|
| **1st** | [7. Hybrid MPC-RL with Learned Adaptation](#7-hybrid-mpc-rl-with-learned-adaptation) | This is the architectural redesign the project is already moving toward. It matches commercial AP system designs (Medtronic 780G, Control-IQ), produces the most defensible engineering artifact, and unlocks every other direction as a modular extension. Start here. |
| **2nd** | [1. Safe RL (Constrained MDP)](#1-safe-rl-constrained-mdp) | Safety is the single most important property of any insulin delivery system. Demonstrating hard hypoglycemia guarantees through constrained optimization is both clinically meaningful and publishable at top ML safety venues. Layer this on top of the hybrid controller. |
| **3rd** | [4. Transformer Sequence Prediction for Glucose Forecasting](#4-transformer-sequence-prediction-for-glucose-forecasting) | A strong glucose forecasting model serves as the internal model for MPC and dramatically improves prediction accuracy over the linearized Hovorka model. The attention-map visualizations are the single most visually compelling demo artifact. |
| **4th** | [18. Explainable AI for Insulin Decisions](#18-explainable-ai-for-insulin-decisions) | Low implementation cost, very high demo value. Counterfactual explanations ("if you had eaten 20g less carbs...") make the project accessible to non-ML audiences and demonstrate clinical relevance. Can be completed in 1--2 weeks alongside other work. |

**Why this ordering.** Direction 7 provides the control architecture that all other
directions plug into. Direction 1 adds the safety guarantee layer that any reviewer
or interviewer will ask about. Direction 4 gives MPC a learned prediction model that
outperforms hand-tuned Hovorka linearizations. Direction 18 wraps the system in
interpretability at minimal cost. Together these four produce a hybrid MPC+RL system
with safety guarantees, learned forecasting, and explainable decisions -- a
portfolio-grade artifact that maps directly to 2--3 publications.

---

## Research Roadmap

```
Month 1            Month 2            Month 3            Month 4            Month 5+
─────────────────  ─────────────────  ─────────────────  ─────────────────  ─────────────────
[7] Hybrid MPC-RL ─────────┐
  (3-4 wk)                 │
                           ├──> [1] Safe RL ──────────┐
[18] XAI ──────┐           │      (3-4 wk)            │
  (1-2 wk)     │           │                          │
               ├───────────┤                          ├──> [5] Dreamer
               │           │                          │      (6-8 wk)
               │    [4] Transformer ──────────┐       │
               │      (3-4 wk)                │       │   OR
               │                              │       │
               │                              ├───────┤  [8] Meta-Learning
               │                              │       │      (6-8 wk)
               │                       [3] Ensemble   │
               │                         MPC (2-3 wk) │   OR
               │                              │       │
               │                              │  [2] Uncertainty-Aware
               │                              │       (4-6 wk)
               │                              │
               └── [10] Offline RL ───────────┘
                     (2-3 wk, parallel)


Legend:
  ────>  sequential dependency (output feeds into next direction)
  Directions without arrows can run in parallel
  Indented directions are optional extensions
```

**Phase 1 (Weeks 1--6): Foundation.** Build the hybrid MPC-RL controller (Direction 7)
and add XAI overlays (Direction 18). These two establish the core architecture and
demo shell.

**Phase 2 (Weeks 4--10): Intelligence.** Train the transformer glucose forecaster
(Direction 4) and plug it into MPC as the internal model. In parallel, implement
ensemble MPC (Direction 3) as a simpler alternative/complement. Optionally begin
offline RL data collection (Direction 10).

**Phase 3 (Weeks 8--14): Safety and Robustness.** Add constrained MDP optimization
(Direction 1) on top of the hybrid controller. This is the publication-critical
phase.

**Phase 4 (Weeks 12+): Advanced Extensions.** Pursue one or two "Very Hard"
directions -- Dreamer-style world models (Direction 5) or meta-learning patient
adaptation (Direction 8) -- for top-venue publication targets.

---

## Research Directions

### 1. Safe RL (Constrained MDP)

**Description.** Reformulate the insulin delivery MDP as a Constrained MDP (CMDP)
where the optimization objective remains maximizing Time-in-Range while enforcing a
hard constraint that the expected time below 54 mg/dL (Level 2 hypoglycemia) stays
below a clinically defined threshold (TBR < 1%). Implement this using Constrained
Policy Optimization (CPO), Reward-Constrained Policy Optimization (RCPO), or
Lagrangian relaxation methods that augment the policy gradient with a learned dual
variable penalizing constraint violations. The constraint is not merely a reward
penalty -- it is a formal optimization constraint with convergence guarantees.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | Critical |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | High |
| **Publication Potential** | Conference (ICML/NeurIPS safety track) |

**Key implementation notes.**
- The cost function `c(s,a) = 1 if BGL < 54 else 0` is non-differentiable; use
  a sigmoid relaxation `sigma((54 - BGL) / tau)` with temperature `tau` for
  gradient-based methods.
- CPO requires computing the Fisher information matrix of the policy -- use
  conjugate gradient approximation for tractability.
- Lagrangian relaxation is simpler: maintain a dual variable `lambda` that is
  updated by dual ascent whenever the constraint is violated in expectation.
  `lambda` effectively converts the hard constraint into an adaptive penalty
  whose weight is learned, not hand-tuned.
- Validate against the unconstrained baseline: the key result is not just TIR
  improvement but the guarantee that TBR stays below threshold across all
  test scenarios, including adversarial meal patterns.

**Relevant references.** Achiam et al. (2017) "Constrained Policy Optimization";
Tessler et al. (2019) "Reward Constrained Policy Optimization"; Stooke et al. (2020)
"Responsive Safety in Reinforcement Learning."

---

### 2. Uncertainty-Aware Control (Bayesian RL)

**Description.** Train an ensemble of N (typically 5--10) neural network policies
that share an architecture but are initialized with different random seeds and
trained on bootstrapped subsets of experience. At inference time, use the disagreement
between ensemble members as a calibrated measure of epistemic uncertainty. When
uncertainty exceeds a threshold, the controller falls back to a conservative MPC
mode with widened safety margins. Exploration during training uses Thompson
sampling -- at each episode, one ensemble member is sampled and followed, producing
natural exploration without explicit noise injection.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 4--6 weeks |
| **Demo Value** | Very High (uncertainty bands are visually compelling) |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Uncertainty bands on glucose forecasts are the visual centerpiece: plot the
  ensemble's mean prediction +/- 2 standard deviations as shaded regions on
  the glucose trajectory chart.
- The MPC fallback trigger should be: if ensemble standard deviation on the
  predicted next-30-min glucose exceeds X mg/dL, switch to conservative mode.
  Tune X on a held-out validation set.
- Training N separate policies is embarrassingly parallel -- launch N
  independent training runs with different seeds and aggregate.
- For computational efficiency at inference, consider using MC-Dropout
  (Gal & Ghahramani, 2016) as a cheaper alternative to full ensembles,
  though calibration is typically worse.
- Integration with Direction 7 (Hybrid MPC-RL): the RL supervisory agent
  becomes the ensemble, and the MPC fallback is the existing MPC layer.

**Relevant references.** Lakshminarayanan et al. (2017) "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles"; Osband et al. (2016)
"Deep Exploration via Bootstrapped DQN."

---

### 3. Ensemble MPC with Online Patient Identification

**Description.** Run 5--10 Model Predictive Controllers in parallel, each
parameterized with a different plausible set of Hovorka patient model parameters
(varying insulin sensitivity SI, endogenous glucose production EGP, carbohydrate
absorption rate, and insulin time constants). At each control step, weight the
insulin recommendations from each MPC instance by how well that instance's
one-step-ahead glucose predictions matched the actual CGM readings over the last
1--2 hours. The result is a Bayesian model-averaging controller that adapts to
patient physiology online without any RL training.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 2--3 weeks |
| **Demo Value** | High |
| **Publication Potential** | Conference (biomedical engineering venues: EMBC, JBHI) |

**Key implementation notes.**
- Initialize the ensemble from the existing patient profiles in
  `configs/profiles/` (insulin-sensitive, insulin-resistant, unstable,
  controlled) plus interpolated combinations.
- Weight update rule: use a softmax over negative prediction errors:
  `w_i = exp(-alpha * MSE_i) / sum(exp(-alpha * MSE_j))` where `MSE_i` is
  model i's mean squared prediction error over the recent window.
- Each MPC instance solves a small quadratic program (QP) at each 5-minute
  step. With 10 instances, this is still real-time feasible -- each QP is
  ~50 decision variables.
- The Hovorka ODE is already implemented in `ap_rl.envs.hovorka_patient` --
  reuse it as each MPC instance's internal model with different parameter
  vectors.
- This direction is a strong standalone baseline against which RL-based
  methods should be compared.

**Relevant references.** Magni et al. (2009) "Model Predictive Control of
Glucose Concentration in Type I Diabetic Patients"; Eren-Oruklu et al. (2009)
"Adaptive System Identification for Estimating Future Glucose Concentrations."

---

### 4. Transformer Sequence Prediction for Glucose Forecasting

**Description.** Train a Transformer encoder on sequences of (glucose, insulin
dose, meal carbohydrates, time-of-day) tuples sampled at 5-minute intervals to
predict future glucose values 30, 60, 90, and 120 minutes ahead. The model ingests
a context window of 6--12 hours of history and outputs a multi-horizon glucose
forecast. Use this trained forecaster as the internal prediction model within MPC,
replacing the linearized Hovorka model with a data-driven learned model that
captures nonlinear meal absorption dynamics and inter-day variability.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | Very High (attention maps show what the model "looks at") |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Generate training data by running the Hovorka simulator across all 200 test
  scenarios in `data/test_scenarios/` with the existing PID and RL controllers,
  producing ~200 x 24h = 4800 hours of paired (input, label) sequences.
- Architecture: 4-layer Transformer encoder with causal masking, 64-dim
  embeddings, 4 attention heads. Input: concatenated (glucose, insulin, carbs,
  sin(time), cos(time)) at each timestep. Output: linear projection to
  4 forecast horizons.
- Attention map visualization is the key demo artifact: extract attention
  weights from the final layer and overlay on the glucose timeline to show
  which past events (meals, insulin doses) the model attends to when making
  each prediction. This is visually striking and clinically interpretable.
- For MPC integration, the transformer replaces the Hovorka ODE in the
  prediction step. MPC optimizes insulin over a 2-hour horizon using the
  transformer's differentiable forward pass.
- Baseline comparison: LSTM, linear AR, and the raw Hovorka ODE model.
  Report RMSE at each forecast horizon.

**Relevant references.** Li et al. (2023) "GluFormer: Transformer-Based
Personalized Glucose Forecasting"; Zhu et al. (2022) "Enhancing Self-Supervised
Blood Glucose Prediction with Transformers."

---

### 5. Latent World Models (Dreamer-style)

**Description.** Learn a compact latent dynamics model of the diabetic patient using
the Dreamer framework: an encoder compresses the high-dimensional patient state
(glucose trajectory, insulin-on-board, meal history) into a low-dimensional latent
vector; a transition model predicts the next latent state given an insulin action; a
decoder reconstructs glucose observations from latent states. Once learned, the
policy is optimized entirely through "imagination" -- rollouts in latent space --
eliminating the need for the explicit Hovorka ODE during planning and enabling
thousands of imagined trajectories per second.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Very Hard |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 6--8 weeks |
| **Demo Value** | Very High |
| **Publication Potential** | Top venue (ICLR/NeurIPS) |

**Key implementation notes.**
- The Dreamer architecture has three components: (1) Recurrent State-Space
  Model (RSSM) for latent dynamics, (2) image/observation encoder/decoder
  (here: glucose sequence encoder), (3) actor-critic operating in latent space.
- Training loop: collect real trajectories from the Hovorka simulator, train
  the world model on reconstruction + KL losses, then train the policy on
  imagined 15-step rollouts in latent space using backpropagation through
  the differentiable world model.
- The latent space should be ~32--64 dimensional. Visualize the learned latent
  space using t-SNE/UMAP colored by patient type and glycemic state -- this
  is a compelling figure for publications.
- Key challenge: the Hovorka model has long-horizon insulin dynamics (insulin
  action peaks 60--90 minutes after injection). The RSSM must capture these
  delays, which requires a sufficiently long context window (at least 2--3
  hours of history in the recurrent state).
- Compare sample efficiency against model-free A2C: Dreamer should require
  10--100x fewer simulator interactions to reach equivalent TIR.

**Relevant references.** Hafner et al. (2020) "Dream to Control: Learning Behaviors
by Latent Imagination" (Dreamer v1); Hafner et al. (2023) "Mastering Diverse Domains
through World Models" (DreamerV3).

---

### 6. Model-Based RL with Differentiable Simulator

**Description.** Re-implement the Hovorka ODE solver using a differentiable
framework (torchdiffeq or diffrax) so that gradients can flow backward through the
entire simulation -- from the glucose outcome, through the ODE integration, back to
the controller parameters. This enables direct gradient-based optimization of the
controller without the high variance of policy gradient estimators. The controller
(whether PID gains, MPC weights, or a neural network) is optimized by
backpropagating the glucose-control loss through the differentiable patient model.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 2--3 weeks |
| **Demo Value** | High |
| **Publication Potential** | Workshop/Conference |

**Key implementation notes.**
- The Hovorka model is a system of 11 coupled ODEs. Translating from the
  existing NumPy implementation in `ap_rl.envs.hovorka_patient` to PyTorch
  tensors is straightforward -- the ODE right-hand-side is algebraic.
- Use `torchdiffeq.odeint_adjoint` for memory-efficient backpropagation
  through long simulations (the adjoint method has O(1) memory in the
  number of ODE steps).
- Optimize a 24-hour glucose trajectory end-to-end: define a loss as
  `L = -TIR + lambda_hypo * TBR + lambda_var * glucose_variance` and
  backpropagate through the entire 288-step (5-min intervals) simulation.
- Gradient clipping is essential -- the ODE can amplify gradients over long
  horizons. Use gradient norm clipping at 1.0.
- This approach is complementary to RL: use differentiable simulation for
  initial controller tuning, then fine-tune with RL for robustness to
  model mismatch.

**Relevant references.** Chen et al. (2018) "Neural Ordinary Differential Equations";
Kidger (2022) "On Neural Differential Equations" (PhD thesis, diffrax).

---

### 7. Hybrid MPC-RL with Learned Adaptation

**Description.** Implement the hierarchical architecture described in the project's
refactor blueprint: MPC provides the base insulin control at 5-minute intervals using
a prediction horizon of 2--4 hours and quadratic cost on glucose deviation + insulin
usage. An RL supervisory agent operates at a slower timescale (every 15--30 minutes)
and adjusts the MPC's tunable parameters -- cost function weights, prediction
horizon length, glucose target setpoint, and aggressiveness level -- based on the
current patient state, recent glucose trends, and time of day. The RL agent learns
*when* to be aggressive and when to be conservative, while MPC handles the
constrained optimization at each step.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | Critical |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | Very High |
| **Publication Potential** | Conference (best approach for this project) |

**Key implementation notes.**
- MPC implementation: use `cvxpy` or `scipy.optimize.minimize` to solve the
  finite-horizon optimal control problem at each 5-minute step. The internal
  model is the linearized Hovorka equations (or the transformer forecaster
  from Direction 4 once available).
- RL action space (what the supervisory agent controls):
  - `target_glucose`: 100--140 mg/dL (lower at night, higher after meals)
  - `aggressiveness`: scalar 0.5--2.0 multiplying the insulin cost weight
  - `prediction_horizon`: 12--48 steps (1--4 hours)
  - `meal_bolus_fraction`: 0.5--1.0 of calculated bolus (partial bolusing)
- RL observation space: recent 2-hour glucose trajectory (downsampled),
  current IOB, time of day, estimated carbs-on-board, meal announcement flag.
- The RL agent acts every 3--6 MPC steps (15--30 minutes), making the MDP
  much lower-dimensional and easier to solve than the current 5-minute
  PID-delta formulation.
- Safety layer sits *below* MPC: clips insulin to `[0, max_bolus]`, suspends
  delivery if predicted glucose < 70 mg/dL within 30 minutes, enforces
  minimum inter-bolus interval.
- This is the architecture used by real commercial AP systems and is the
  recommended core for all other directions to build upon.

**Relevant references.** Hovorka et al. (2004) "Nonlinear model predictive control
of glucose concentration in subjects with type 1 diabetes"; Shi et al. (2019)
"Adaptive Personalized Prior-Knowledge-Informed Model Predictive Control for
Type 1 Diabetes."

---

### 8. Adaptive Patient Embeddings (Meta-Learning)

**Description.** Use Model-Agnostic Meta-Learning (MAML) or learned patient
embeddings to create a controller that can adapt to a new, previously unseen patient
within 2--3 hours of CGM data (approximately 24--36 glucose readings at 5-minute
intervals). In the meta-RL formulation, each patient parameterization is treated as a
separate "environment." The meta-learner finds an initialization of the policy
network that is maximally amenable to rapid fine-tuning -- a few gradient steps on
data from a new patient produce a patient-specific controller without catastrophic
forgetting of general control knowledge.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Very Hard |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 6--8 weeks |
| **Demo Value** | High |
| **Publication Potential** | Top venue |

**Key implementation notes.**
- Meta-training: sample patient parameters from the population distribution
  defined in `configs/profiles/`. Each meta-episode: (1) sample a patient,
  (2) collect 2--3 hours of data with the current meta-policy, (3) take K
  inner-loop gradient steps to adapt, (4) evaluate adapted policy on the
  remaining 21--22 hours, (5) update the meta-parameters using the outer
  loss.
- Alternative to MAML: learn a patient embedding vector `z_patient` (8--16
  dimensional) that is inferred from recent glucose data using an encoder
  network and concatenated to the policy's input. The encoder is trained
  end-to-end with the policy. At deployment, the encoder runs continuously
  on a sliding window of data, automatically adapting `z_patient` as patient
  physiology shifts.
- The patient embedding approach is simpler to implement than MAML and avoids
  second-order gradients. It also enables visualization: plot the learned
  embedding space and observe clustering by patient type.
- Key metric: adaptation speed. Measure TIR at 1h, 2h, 4h, 8h, and 24h
  after encountering a new patient, compared to a non-adaptive baseline.
- Generate a diverse meta-training population by interpolating/extrapolating
  between existing patient profiles (Direction 14 can provide additional
  synthetic patients).

**Relevant references.** Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast
Adaptation of Deep Networks"; Rakelly et al. (2019) "Efficient Off-Policy
Meta-Reinforcement Learning via Probabilistic Context Variables" (PEARL).

---

### 9. Continual Learning with Catastrophic Forgetting Prevention

**Description.** Apply continual learning techniques to prevent the RL policy from
forgetting how to treat previously seen patient types when fine-tuned on a new one.
Use Elastic Weight Consolidation (EWC), which adds a quadratic penalty on weight
changes proportional to Fisher information (protecting parameters important for
previous patients), or PackNet, which progressively prunes and freezes subnetworks
for each patient while leaving free capacity for new ones. The goal is a single
policy network that handles the full patient population without per-patient
retraining.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | Medium |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | Medium |
| **Publication Potential** | Workshop |

**Key implementation notes.**
- EWC implementation: after training on patient A, compute the Fisher
  information matrix diagonal `F_A` for each network parameter. When
  training on patient B, add the penalty
  `(lambda/2) * sum(F_A * (theta - theta_A)^2)` to the loss. This prevents
  the weights most important for patient A from changing.
- PackNet alternative: train on patient A, prune 75% of weights by
  magnitude, freeze the remaining 25%, then train the freed weights on
  patient B. This gives hard isolation between patients at the cost of
  network capacity.
- Evaluation protocol: train sequentially on patients [insulin-sensitive,
  insulin-resistant, unstable, controlled]. After each, evaluate TIR on
  ALL previous patients. Report the "backward transfer" metric (performance
  on patient A after training on patient D).
- The current A2C policy network (`ap_rl.agents.diabetes_a2c_actor`) is
  small enough (2 hidden layers, 64 units) that EWC should work without
  capacity issues. For PackNet, consider increasing to 128 or 256 units.

**Relevant references.** Kirkpatrick et al. (2017) "Overcoming catastrophic
forgetting in neural networks" (EWC); Mallya & Lazebnik (2018) "PackNet:
Adding Multiple Tasks to a Single Network by Iterative Pruning."

---

### 10. Offline RL from Clinical Datasets

**Description.** Train a control policy using batch/offline RL algorithms --
Conservative Q-Learning (CQL), Implicit Q-Learning (IQL), or Decision Transformer
-- on a pre-collected dataset of (state, action, reward, next_state) transitions
without any further interaction with the simulator during training. This
demonstrates a "safe deployment" paradigm: the policy is learned entirely from
historical data, never explores dangerously, and can be evaluated on held-out
trajectories before any deployment. The approach is directly applicable to real
clinical data if it becomes available.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 2--3 weeks |
| **Demo Value** | High |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Dataset generation: run the current PID baseline and A2C controller across
  all 200 test scenarios, collecting full trajectories. Add suboptimal
  policies (random PID gains, degraded A2C) for diversity. Target: 1M+
  transitions.
- CQL adds a conservative penalty to Q-values for out-of-distribution
  actions, preventing the policy from selecting actions not well-supported
  by the dataset. This is critical for insulin delivery where unseen
  aggressive dosing could cause hypoglycemia.
- Decision Transformer frames control as sequence modeling: condition on
  desired return (target TIR) and generate actions autoregressively. This
  enables return-conditioned control at test time -- ask for 90% TIR and
  the policy adjusts.
- Key comparison: offline RL policy vs. the online A2C agent. Report both
  TIR and the distributional shift metric (how far offline actions deviate
  from the dataset distribution).
- The "never explores dangerously" property is the main selling point for
  clinical audiences. Emphasize this in any publication.

**Relevant references.** Kumar et al. (2020) "Conservative Q-Learning for Offline
Reinforcement Learning" (CQL); Kostrikov et al. (2022) "Offline Reinforcement
Learning with Implicit Q-Learning" (IQL); Chen et al. (2021) "Decision Transformer:
Reinforcement Learning via Sequence Modeling."

---

### 11. Multi-Objective RL (Pareto-Optimal Control)

**Description.** Simultaneously optimize three competing objectives -- maximize
Time-in-Range (TIR 70--180), minimize hypoglycemia risk (time below 54 mg/dL), and
minimize total daily insulin dose -- using multi-objective RL. Instead of scalarizing
these into a single reward (which requires hand-tuning weights), learn the entire
Pareto frontier of non-dominated policies. At deployment, the clinician or patient
selects their preferred trade-off point on an interactive Pareto frontier
visualization.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | Medium |
| **Implementation Complexity** | 4--5 weeks |
| **Demo Value** | Very High (interactive Pareto visualization) |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Use Pareto Conditioned Networks (PCN) or Multi-Objective Maximum a
  Posteriori Policy Optimization (MO-MPO). PCN conditions the policy on
  a desired objective weighting vector `w = (w_TIR, w_hypo, w_insulin)`,
  sampled uniformly during training, producing a single network that
  represents the entire Pareto frontier.
- The interactive demo is the highlight: a Streamlit/Plotly scatter plot of
  the Pareto frontier where clicking a point runs the corresponding policy
  and shows the glucose trajectory. The user drags a slider between "minimize
  hypo risk" and "maximize TIR" and watches the glucose trace change in
  real time.
- Three-objective Pareto fronts are harder to visualize than two-objective.
  Consider presenting 2D projections (TIR vs. TBR, TIR vs. insulin) as
  well as a 3D interactive scatter.
- Reward vector: `r = (r_TIR, r_hypo, r_insulin)` where each component is
  computed at each timestep. `r_TIR = 1 if 70 <= BGL <= 180 else 0`,
  `r_hypo = -(max(0, 54 - BGL))^2`, `r_insulin = -insulin_dose`.

**Relevant references.** Lu et al. (2023) "Multi-Objective Reinforcement Learning:
Convexity, Stationarity and Pareto Optimality"; Abdolmaleki et al. (2020)
"A Distributional View on Multi-Objective Policy Optimization."

---

### 12. Probabilistic Digital Twins

**Description.** Build a per-patient probabilistic model using either a Gaussian
Process (GP) or Neural ODE that captures both the deterministic glucose dynamics and
the uncertainty around predictions. The GP/Neural ODE is trained on an individual
patient's historical data and produces glucose forecasts with credible intervals
(e.g., "glucose will be 145 +/- 18 mg/dL in 60 minutes with 95% confidence"). This
personalized "digital twin" enables risk-aware control: MPC can optimize against
the worst-case bound of the credible interval rather than the point estimate.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 4--6 weeks |
| **Demo Value** | Very High |
| **Publication Potential** | Journal |

**Key implementation notes.**
- GP approach: use a Sparse GP (inducing points) with an RBF + periodic
  kernel to capture circadian glucose patterns. Input: (time_of_day,
  recent_glucose_window, IOB, COB). Output: glucose at t+30, t+60, t+90,
  t+120 with posterior variance.
- Neural ODE approach: parameterize the Hovorka ODE's uncertain parameters
  (SI, EGP, k_abs) as distributions. Use variational inference or ensembles
  to propagate uncertainty through the ODE integration.
- The credible interval visualization is the main demo artifact: plot the
  glucose trajectory with shaded 50%/80%/95% credible bands. Overlay the
  actual CGM readings. Narrowing bands indicate the model is confident;
  widening bands indicate physiological novelty.
- For MPC integration: optimize against the upper bound of the 95% interval
  for hypoglycemia (conservative) and the point estimate for hyperglycemia
  (aggressive). This asymmetric risk treatment matches clinical practice.
- Per-patient models need at least 3--7 days of data to calibrate. Evaluate
  on held-out days from the same patient.

**Relevant references.** Plis et al. (2014) "A Machine Learning Approach to
Predicting Blood Glucose Levels for Diabetes Management"; Kidger et al. (2020)
"Neural SDEs as Infinite-Dimensional GANs" (for stochastic Neural ODEs).

---

### 13. Diffusion Models for Scenario Generation

**Description.** Train a conditional diffusion model to generate realistic 24-hour
glucose-relevant scenarios -- meal timing and sizes, exercise events, stress
responses, and dawn phenomenon patterns -- from a small set of conditioning
variables (patient type, activity level, dietary preference, day of week). The
generated scenarios serve as unlimited training data for the RL controller, replacing
the fixed set of 200 test scenarios in `data/test_scenarios/` with arbitrarily
diverse and physiologically plausible daily patterns.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | Medium |
| **Implementation Complexity** | 4--5 weeks |
| **Demo Value** | High |
| **Publication Potential** | Workshop/Conference |

**Key implementation notes.**
- Model: 1D U-Net denoising network operating on 288-step (24-hour at
  5-min) sequences. Each timestep has channels: (meal_carbs, exercise_intensity,
  stress_level). Conditioning via cross-attention on (patient_type_embedding,
  activity_level, dietary_preference).
- Training data: generate the "ground truth" scenarios by running all 200
  existing test cases through the Hovorka simulator and recording the
  exogenous inputs. Augment with hand-crafted edge cases (dawn phenomenon,
  late-night snacking, double meals).
- Evaluation: compare the distribution of generated scenario statistics
  (total daily carbs, meal timing entropy, max single meal size) against
  the real scenario distribution using the Wasserstein distance.
- The generated scenarios should include "hard" cases that stress-test the
  controller -- large unannounced meals, exercise during insulin peak,
  consecutive high-carb meals.
- Integration: use generated scenarios to augment the training set for any
  other direction, particularly Directions 1, 7, and 10.

**Relevant references.** Ho et al. (2020) "Denoising Diffusion Probabilistic Models";
Tashiro et al. (2021) "CSDI: Conditional Score-based Diffusion Models for
Probabilistic Time Series Imputation."

---

### 14. Synthetic Patient Generation via GANs/VAEs

**Description.** Learn the joint distribution of Hovorka patient model parameters
(insulin sensitivity, endogenous glucose production, carb absorption rate, body
weight, and other physiological constants) from a population and generate realistic
new virtual patients by sampling from this learned distribution. A GAN or VAE
trained on the existing patient parameter space produces physiologically plausible
parameter vectors that can be directly plugged into the Hovorka ODE simulator to
create new virtual patients for training and evaluation.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | Medium |
| **Implementation Complexity** | 2--3 weeks |
| **Demo Value** | Medium |
| **Publication Potential** | Workshop |

**Key implementation notes.**
- The Hovorka model has approximately 15--20 free parameters. Collect a
  "population" by: (1) the 4 existing profiles in `configs/profiles/`,
  (2) literature-reported parameter ranges from Hovorka et al. (2004) and
  Wilinska et al. (2010), (3) Latin Hypercube Sampling within physiologically
  valid bounds.
- VAE is preferred over GAN for this low-dimensional setting (15--20 params):
  the VAE latent space is smooth and interpolable, and training is stable.
  Use a 4-dimensional latent space.
- Validation: generate 100 synthetic patients, simulate each with the current
  PID baseline, and verify that the resulting glucose distributions match
  published population statistics (mean TIR, HbA1c distribution, hypo
  frequency).
- Reject sampling: discard generated patients whose parameters fall outside
  physiological bounds (e.g., negative insulin sensitivity, body weight < 30 kg).
- This direction primarily supports other directions (8, 9, 10, 13) by
  expanding the patient population for training and evaluation.

**Relevant references.** Xie et al. (2019) "Simglucose v0.2.1"; Man et al. (2014)
"The UVA/Padova Type 1 Diabetes Simulator" (for population parameter distributions).

---

### 15. Hierarchical RL with Temporal Abstraction

**Description.** Implement the Options framework for hierarchical RL: a high-level
RL agent selects among macro-actions ("deliver correction bolus," "increase basal by
20%," "decrease basal by 20%," "maintain current rate," "suspend delivery") at
15-minute intervals. A low-level MPC controller executes each selected macro-action
optimally over the next 15 minutes, respecting physiological constraints. This
temporal abstraction reduces the effective MDP horizon by 3x, makes the high-level
policy interpretable (it makes decisions a clinician would recognize), and decouples
strategic decisions from tactical execution.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Medium |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | High |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Define 5--7 discrete options (macro-actions):
  - `MAINTAIN`: keep current basal rate unchanged.
  - `INCREASE_BASAL_SMALL`: +10% basal for 15 min.
  - `INCREASE_BASAL_LARGE`: +25% basal for 15 min.
  - `DECREASE_BASAL`: -25% basal for 15 min.
  - `SUSPEND`: 0 insulin for 15 min (hypo prevention).
  - `CORRECTION_BOLUS`: deliver a calculated correction dose.
  - `SUPER_BOLUS`: borrow future basal into an upfront bolus (aggressive).
- Each option has a termination condition (fixed 15-minute duration for
  simplicity, or learned termination for the advanced version).
- The high-level policy is a small discrete-action DQN or PPO agent.
  The observation includes: 2-hour glucose history, current IOB, COB
  estimate, time of day, and current basal rate.
- The low-level MPC within each option solves for the optimal insulin
  trajectory to achieve the macro-action's intent (e.g., "increase basal
  by 20%" is implemented by MPC targeting a basal rate 1.2x the patient's
  programmed rate).
- Visualization: a timeline showing which macro-action is active at each
  point, overlaid on the glucose trace. This is immediately interpretable
  by clinicians.

**Relevant references.** Sutton, Precup & Singh (1999) "Between MDPs and
Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning";
Bacon, Harb & Precup (2017) "The Option-Critic Architecture."

---

### 16. Imitation Learning from Commercial AP Data

**Description.** Collect published clinical trial traces or simulator-derived
control traces from commercial artificial pancreas systems (Medtronic 780G,
Tandem Control-IQ, Omnipod 5) and use behavioral cloning followed by DAgger
(Dataset Aggregation) to train a neural network policy that replicates their
control behavior. The resulting policy serves as both a strong baseline and a
demonstration that learning-based control can match purpose-built commercial
systems.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Easy (if data available) |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 1--2 weeks |
| **Demo Value** | Very High (compare against real commercial systems) |
| **Publication Potential** | Conference (if data licensing allows) |

**Key implementation notes.**
- Data sources: OpenAPS Data Commons (self-reported open-source AP data),
  Tidepool Big Data Donation Project (anonymized CGM + pump data), published
  clinical trial supplementary data.
- Behavioral cloning baseline: supervised learning mapping
  (glucose_history, IOB, COB, time_of_day) -> insulin_dose. Use the same
  network architecture as the A2C actor in `ap_rl.agents.diabetes_a2c_actor`.
- DAgger improvement: after behavioral cloning, run the learned policy in
  the Hovorka simulator. At states where the policy diverges from the expert,
  query the expert (or a proxy) for the correct action and add to the
  dataset. Repeat for 3--5 iterations.
- If real clinical data is unavailable, simulate "commercial AP" behavior
  by implementing a well-tuned MPC controller (from Direction 7) as the
  expert, then clone it. This still demonstrates the imitation learning
  pipeline.
- Key demo figure: overlay your learned policy's glucose trace against the
  commercial system's trace on the same patient scenario. Close tracking
  demonstrates the method works; divergence points highlight where your
  system could improve.

**Relevant references.** Ross, Gordon & Bagnell (2011) "A Reduction of Imitation
Learning and Structured Prediction to No-Regret Online Learning" (DAgger);
Lewis et al. (2016) "Real-World Use of Open Source Artificial Pancreas Systems."

---

### 17. Adversarial Robustness Testing

**Description.** Train an adversarial agent to find worst-case disturbance
sequences -- meal timings, exercise events, sensor noise patterns, and insulin
absorption variations -- that maximally degrade the controller's performance. The
controller and adversary are trained in a minimax game: the controller minimizes
glucose excursions while the adversary maximizes them. The resulting robust
controller is certified against adversarial disturbances, and the adversary itself
becomes a valuable tool for stress-testing any controller.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Hard |
| **Expected Impact on Glucose Control** | High |
| **Implementation Complexity** | 3--4 weeks |
| **Demo Value** | High |
| **Publication Potential** | Conference |

**Key implementation notes.**
- Adversary action space: at each decision point, the adversary controls
  (1) whether a surprise meal occurs (0 or 1), (2) meal size (0--120g carbs),
  (3) exercise intensity (0--1), (4) sensor noise magnitude, (5) insulin
  absorption rate multiplier (0.7--1.3, simulating subcutaneous variability).
- Training: alternate between adversary and controller updates (RARL --
  Robust Adversarial RL). Use PPO for both. The adversary's reward is the
  negative of the controller's reward.
- Constraint on the adversary: the adversary must produce physiologically
  plausible scenarios. Enforce constraints like total daily carbs in
  [150, 400]g, max single meal < 120g, max 2 exercise sessions per day.
  Without constraints, the adversary will produce degenerate scenarios
  (infinite meals, constant exercise).
- The discovered worst-case scenarios are themselves a valuable output:
  they reveal the controller's failure modes and guide further development.
  Catalog the top-10 adversarial scenarios.
- Key metric: performance under adversarial scenarios vs. random scenarios.
  The robust controller should show smaller degradation.

**Relevant references.** Pinto et al. (2017) "Robust Adversarial Reinforcement
Learning"; Tessler et al. (2019) "Action Robust Reinforcement Learning and
Applications in Continuous Control."

---

### 18. Explainable AI for Insulin Decisions

**Description.** Add interpretability layers to the insulin delivery controller using
SHAP values for feature attribution, attention visualization (if using transformer
components), and counterfactual explanations that answer questions like "if you had
eaten 20g fewer carbs, glucose would have peaked at 160 instead of 200 mg/dL" or
"the controller increased basal because glucose rate-of-change exceeded +2 mg/dL/min
for 15 minutes." These explanations make the controller's decisions transparent to
clinicians, patients, and reviewers.

| Attribute | Value |
|-----------|-------|
| **Difficulty** | Easy |
| **Expected Impact on Glucose Control** | Medium |
| **Implementation Complexity** | 1--2 weeks |
| **Demo Value** | Very High |
| **Publication Potential** | Workshop |

**Key implementation notes.**
- SHAP values: use KernelSHAP on the A2C actor network (or the RL
  supervisory agent from Direction 7) to compute per-feature importance
  for each insulin decision. The 19-dimensional observation space is small
  enough for exact SHAP computation.
- Feature importance visualization: a waterfall chart showing how each
  observation feature (current glucose, glucose rate, IOB, time of day,
  Kp, Ki, Kd, etc.) contributes to the insulin decision at a selected
  timepoint. Overlay on the glucose timeline in the Streamlit demo.
- Counterfactual explanations: for a given timepoint, perturb one input
  variable (e.g., meal size, exercise timing) and re-run the simulation
  from that point. Display the "what-if" glucose trajectory alongside
  the actual trajectory.
- Integrate into the existing Streamlit demo (`app/app.py`): add a tab
  or sidebar panel for "Explain this decision" that shows SHAP values
  and counterfactuals when the user clicks on any point in the glucose
  chart.
- Low implementation effort because SHAP and counterfactual libraries
  (shap, alibi, DiCE) handle the heavy lifting. The main work is
  integration and visualization.

**Relevant references.** Lundberg & Lee (2017) "A Unified Approach to Interpreting
Model Predictions" (SHAP); Wachter et al. (2018) "Counterfactual Explanations
without Opening the Black Box."

---

## Summary Table

| # | Direction | Difficulty | Impact | Complexity | Demo Value | Publication |
|---|-----------|-----------|--------|------------|------------|-------------|
| 1 | Safe RL (Constrained MDP) | Hard | Critical | 3--4 weeks | High | Conference (ICML/NeurIPS) |
| 2 | Uncertainty-Aware Control | Hard | High | 4--6 weeks | Very High | Conference |
| 3 | Ensemble MPC + Online ID | Medium | High | 2--3 weeks | High | Conference (biomed) |
| 4 | Transformer Glucose Forecasting | Medium | High | 3--4 weeks | Very High | Conference |
| 5 | Latent World Models (Dreamer) | Very Hard | High | 6--8 weeks | Very High | Top venue |
| 6 | Differentiable Simulator | Medium | High | 2--3 weeks | High | Workshop/Conference |
| 7 | Hybrid MPC-RL | Medium | Critical | 3--4 weeks | Very High | Conference |
| 8 | Meta-Learning Patient Adapt. | Very Hard | High | 6--8 weeks | High | Top venue |
| 9 | Continual Learning | Hard | Medium | 3--4 weeks | Medium | Workshop |
| 10 | Offline RL | Medium | High | 2--3 weeks | High | Conference |
| 11 | Multi-Objective RL (Pareto) | Hard | Medium | 4--5 weeks | Very High | Conference |
| 12 | Probabilistic Digital Twins | Hard | High | 4--6 weeks | Very High | Journal |
| 13 | Diffusion Scenario Generation | Hard | Medium | 4--5 weeks | High | Workshop/Conference |
| 14 | Synthetic Patient Gen (GAN/VAE) | Medium | Medium | 2--3 weeks | Medium | Workshop |
| 15 | Hierarchical RL (Options) | Medium | High | 3--4 weeks | High | Conference |
| 16 | Imitation Learning | Easy | High | 1--2 weeks | Very High | Conference |
| 17 | Adversarial Robustness | Hard | High | 3--4 weeks | High | Conference |
| 18 | Explainable AI | Easy | Medium | 1--2 weeks | Very High | Workshop |

---

## Dependency Graph

Some directions build on others. The following shows which directions are
prerequisites, enablers, or natural follow-ons:

```
                    [7] Hybrid MPC-RL (foundation)
                   /    |    |    \         \
                  /     |    |     \         \
                 v      v    v      v         v
             [1]     [4]  [15]   [2]       [18]
             Safe    Tfmr  Hier  Uncert.    XAI
             RL      Pred  RL    Aware
              |       |      |
              v       v      v
            [17]    [12]   [10]
            Adv.    Dig.   Offline
            Rob.    Twin   RL
                      |
                      v
                    [5] Dreamer (uses learned dynamics)

     Independent enablers (can start anytime):
     [3] Ensemble MPC    [6] Diff. Simulator    [14] Synth. Patients
     [13] Diffusion Scenarios    [16] Imitation Learning
     [11] Multi-Obj RL    [9] Continual Learning

     [8] Meta-Learning Patient Adapt. depends on [14] for population diversity
```

---

## Quick-Start Guidance by Goal

**"I want the strongest possible demo in 4 weeks."**
Start with [7] Hybrid MPC-RL + [18] XAI + [16] Imitation Learning. This gives you
a commercial-architecture controller with explainable decisions and a comparison
against real AP system behavior.

**"I want to publish at a top ML venue."**
Pursue [5] Latent World Models or [8] Meta-Learning Patient Adaptation, both of
which are novel enough for ICLR/NeurIPS. Prerequisite: [7] Hybrid MPC-RL as the
base controller.

**"I want to publish at a clinical/biomedical venue."**
Combine [7] Hybrid MPC-RL + [1] Safe RL + [3] Ensemble MPC. Clinical reviewers
care about safety guarantees, physiological plausibility, and comparison against
established methods.

**"I want maximum portfolio impact with minimum time investment."**
Do [7] Hybrid MPC-RL (3--4 weeks) + [18] XAI (1--2 weeks) + [4] Transformer
Forecasting (3--4 weeks). The attention-map visualizations from the transformer and
SHAP explanations from XAI produce visually striking portfolio artifacts. Total:
~8--10 weeks for three publishable components.

**"I want to demonstrate safety for regulatory/clinical audiences."**
Combine [1] Safe RL + [7] Hybrid MPC-RL + [17] Adversarial Robustness + [12]
Probabilistic Digital Twins. This produces a controller with formal safety
constraints, stress-tested against adversarial scenarios, with uncertainty-quantified
predictions.
