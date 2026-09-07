# Hovorka Base Model and Project Extensions Contract

## Purpose

Make the simulator explicit about the boundary between the published 2004
Hovorka model and this project's two research extensions: circadian endogenous
glucose production (EGP) and exercise-driven insulin sensitivity. This is a
research simulator contract, not a clinical validation or treatment protocol.

## Model layers

1. **Base Hovorka 2004:** the ten-state glucose, insulin, and meal model in
   Hovorka and Wilinska (2004), equations (1)-(7), tables 1-2.
2. **Circadian EGP extension:** replace the base `EGP_0` with
   `EGP_0 * (1 + A_EGP * sin(2*pi*(t - phi_EGP)/1440))` before applying the
   existing `(1 - x3)` insulin suppression.
3. **Exercise sensitivity extension:** calculate the project-specific
   dimensionless `F(t)` from the documented exponential rise/decay equations
   and multiply the three insulin-action activation terms:
   `dx_i = F(t) * k_bi * I - k_ai * x_i`, for `i = 1, 2, 3`.

Neither extension is attributed to the 2004 paper. Setting `A_EGP = 0` and
`F_peak = 1` must reproduce the base-model RHS exactly.

## Internal unit contract

| Quantity | Internal unit | Boundary rule |
|---|---|---|
| Time `t` | min | All ODE rate constants use `min^-1` or the matching per-minute unit. |
| Insulin infusion `u` | U/min | Public runner inputs in U/h divide by 60 at the patient boundary. |
| External glucose input `u_g_g_min` | g/min | The RHS divides by `0.180182 g/mmol` at the gut boundary. |
| Insulin compartments `S1`, `S2` | U | They integrate `u` over minutes. |
| Plasma insulin `I` | U/L | `I_mU_L = 1000 * I_U_L` when comparing against the paper. |
| Glucose `Q1`, `Q2`, `D1`, `D2` | mmol | Meals in grams divide by 0.180182 g/mmol at the gut boundary. |
| Glucose concentration `G` | mmol/L | Convert to mg/dL only in public observations. |
| Body weight | kg | `V_I`, `V_G`, `F_01`, and `EGP_0` scale at parameter packing. |

Because the implementation keeps `I` in U/L, the effective activation values
stored in the runtime are the paper's `k_b` values multiplied by 1000:

| Parameter | Paper sensitivity form | Paper `k_b` for mU/L | Runtime `k_b` for U/L |
|---|---:|---:|---:|
| `k_b1` | `S_IT = 51.2e-4` | `0.006 * 0.00512 = 0.00003072` | `0.03072` |
| `k_b2` | `S_ID = 8.2e-4` | `0.06 * 0.00082 = 0.0000492` | `0.0492` |
| `k_b3` | `S_IE = 520e-4` | `0.03 * 0.052 = 0.00156` | `1.56` |

## Canonical base parameters

The canonical base profile uses `k_12=0.066`, `k_a1=0.006`, `k_a2=0.06`,
`k_a3=0.03`, `k_e=0.138`, `V_I=0.12 L/kg`, `V_G=0.16 L/kg`, `A_G=0.8`,
`t_max,G=40 min`, `EGP_0=0.0161 mmol/kg/min`, `F_01=0.0097 mmol/kg/min`,
and `t_max,I=55 min`. It restores the base low-glucose `F_01` correction below
`4.5 mmol/L` and renal clearance above `9 mmol/L`.

The existing `DEFAULT_PATIENT_PARAMS` values are retained as a complete named
legacy profile, not silently discarded. Existing synthetic profiles are not
claimed to be paper-derived and must be rebased deliberately after the base
contract is executable.

## Extension parameters

`F_peak` is dimensionless. `K_rise` and `K_decay` are `min^-1`; `t_rise` and
`t_decay` are minutes. The source documents establish the equations but do not
state the intended numeric unit for the existing literal `K_rise = 5.0`.
The implementation ruling is to interpret that literal as `5 h^-1` and store
`K_rise = 5 / 60 min^-1` in the project exercise profile. This is an explicit
calibration assumption, not a claim about the source; replace that one profile
value if recovered Amesim evidence identifies a different unit.

An empty patient configuration uses `F_peak = 1`, `K_rise = 5 / 60 min^-1`,
and `K_decay = 0.01 min^-1`, so the exercise extension is disabled by default.

## Controller IOB accounting

Controller IOB is a separate simulator-only signal driven exclusively by the
insulin rate that passes the safety layer. It uses the remaining mass of two
equal first-order stages (an Erlang-2 curve), sampled once per minute and
normalized to zero at the configured cutoff. The time constant is one quarter
of `iob_duration_min`, placing peak action near 60 minutes with the default
research assumption `iob_duration_min: 240`. The four-hour duration and curve
shape are transparent project assumptions, not values from Hovorka 2004 and
not a clinically validated dosing model.

The Hovorka absorption states `S1 + S2` remain available as
`sc_depot_insulin_u` for diagnostics. They are not reported as controller IOB.

## Non-goals

- Do not make clinical-safety, treatment, or physiological-validity claims.
- Do not retrain, compare, or publish existing RL checkpoints after model
  changes; they are tied to the old simulator dynamics.
- Do not start the custom frontend until the model and service contracts are
  stable.
