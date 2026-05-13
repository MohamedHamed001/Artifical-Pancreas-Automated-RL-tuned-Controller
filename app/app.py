"""Streamlit digital-twin demo for the Artificial Pancreas RL controller.

Thin orchestration layer that re-uses :class:`ap_rl.envs.DiabetesPIDEnv`
for both baseline (zero-delta PID) and RL (loaded actor) rollouts.

Run with::

    streamlit run app/app.py

Disclaimers shown in the UI: this is a **research / demo simulator**.
The synthetic patient profiles are not clinical patient models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make ``ap_rl`` importable when running directly from a fresh clone
# without ``pip install -e .`` (Streamlit Cloud convenience).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ap_rl.envs import DiabetesPIDEnv
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.envs.profile_loader import PatientProfile, list_profiles, load_profile
from ap_rl.envs.scenario_builder import build_scenario
from ap_rl.runtime.rollout import (
    EpisodeRecord,
    load_actor_from_checkpoints,
    run_episode,
)
from ap_rl.utils.config import load_yaml
from ap_rl.utils.checkpoint_filenames import ACTOR_BEST
from ap_rl.utils.paths import checkpoints_dir, configs_dir
from ap_rl.utils.seed import set_global_seed


st.set_page_config(
    page_title="AP-RL Digital Twin",
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _available_profiles() -> list[str]:
    profiles = list_profiles()
    return profiles or ["controlled"]


@st.cache_data(show_spinner=False)
def _meal_templates() -> dict[str, Path]:
    meals_dir = configs_dir() / "meals"
    if not meals_dir.exists():
        return {}
    return {p.stem: p for p in sorted(meals_dir.glob("*.yaml"))}


@st.cache_data(show_spinner=False)
def _demo_defaults() -> dict:
    demo_path = configs_dir() / "demo.yaml"
    if not demo_path.exists():
        return {}
    return load_yaml(demo_path)


@st.cache_resource(show_spinner=False)
def _load_actor_cached() -> object | None:
    """Cache the actor between reruns. ``None`` when no weights present."""
    return load_actor_from_checkpoints()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _shade_meal_markers(fig: go.Figure, meals: list[dict]) -> None:
    for meal in meals:
        fig.add_vline(
            x=meal["time"],
            line_width=1,
            line_dash="dot",
            line_color="rgba(224, 123, 0, 0.45)",
        )


def _shade_exercise_spans(fig: go.Figure, exercise: list[dict]) -> None:
    start: Optional[float] = None
    for ev in exercise:
        if ev["active"] == 1 and start is None:
            start = ev["time"]
        elif ev["active"] == 0 and start is not None:
            fig.add_vrect(
                x0=start,
                x1=ev["time"],
                fillcolor="rgba(155, 89, 182, 0.10)",
                line_width=0,
                annotation_text="exercise",
                annotation_position="top left",
            )
            start = None


def _heuristic_bg_preview(
    glucose: np.ndarray, horizon_min: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Linear extrapolation from the last two BGL samples.

    This is intentionally a trivial preview (slope of the last two
    samples) - explicitly labelled in the UI as ``heuristic preview``
    so users do not mistake it for a learned forecast.
    """
    if glucose.size < 2:
        return np.array([]), np.array([])
    slope = glucose[-1] - glucose[-2]
    future_steps = np.arange(1, horizon_min + 1)
    preview = glucose[-1] + slope * future_steps
    return future_steps, preview


def build_glucose_figure(
    record: EpisodeRecord,
    target_band: tuple[float, float],
    *,
    show_preview: bool,
    horizon: int,
) -> go.Figure:
    times = np.asarray(record.times[:horizon])
    glucose = np.asarray(record.glucose[:horizon])
    fig = go.Figure()

    fig.add_hrect(
        y0=target_band[0],
        y1=target_band[1],
        fillcolor="rgba(134, 209, 138, 0.18)",
        line_width=0,
        annotation_text=f"Target {target_band[0]:.0f}-{target_band[1]:.0f}",
        annotation_position="top left",
    )
    fig.add_hline(
        y=record.target_glucose,
        line_dash="dash",
        line_color="rgba(42, 127, 58, 0.8)",
        annotation_text=f"Target {record.target_glucose:.0f}",
        annotation_position="bottom right",
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=glucose,
            mode="lines",
            name=f"BGL ({record.controller})",
            line=dict(color="#1f4ea1", width=2.2),
        )
    )

    if show_preview and glucose.size >= 2:
        future_steps, preview = _heuristic_bg_preview(glucose, horizon_min=30)
        future_times = times[-1] + future_steps
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=preview,
                mode="lines",
                name="Heuristic preview (linear)",
                line=dict(color="#7f8c8d", dash="dot", width=1.5),
            )
        )

    _shade_meal_markers(fig, record.meals)
    _shade_exercise_spans(fig, record.exercise)

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (min)",
        yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def build_insulin_figure(record: EpisodeRecord, horizon: int) -> go.Figure:
    times = np.asarray(record.times[:horizon])
    insulin = np.asarray(record.insulin[:horizon])
    bolus = np.asarray(record.bolus[:horizon])
    basal = np.asarray(record.basal[:horizon])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=insulin,
            mode="lines",
            name="Total insulin (U/h)",
            line=dict(color="#c0392b", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=basal,
            mode="lines",
            name="Basal",
            line=dict(color="#7f8c8d", width=1.4, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bolus,
            mode="lines",
            name="Bolus",
            line=dict(color="#e67e22", width=1.4, dash="dot"),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (min)",
        yaxis_title="Insulin (U/h)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def build_glucose_insulin_twin_figure(
    record: EpisodeRecord,
    target_band: tuple[float, float],
    *,
    show_preview: bool,
    horizon: int,
) -> go.Figure:
    """Single Plotly figure: glucose on primary y, total insulin on secondary y."""
    times = np.asarray(record.times[:horizon])
    glucose = np.asarray(record.glucose[:horizon])
    insulin = np.asarray(record.insulin[:horizon])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_hrect(
        y0=target_band[0],
        y1=target_band[1],
        fillcolor="rgba(134, 209, 138, 0.18)",
        line_width=0,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=glucose,
            mode="lines",
            name=f"BGL ({record.controller})",
            line=dict(color="#1f4ea1", width=2.2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=insulin,
            mode="lines",
            name="Total insulin (U/h)",
            line=dict(color="#c0392b", width=2),
        ),
        secondary_y=True,
    )

    if show_preview and glucose.size >= 2:
        future_steps, preview = _heuristic_bg_preview(glucose, horizon_min=30)
        future_times = times[-1] + future_steps
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=preview,
                mode="lines",
                name="Heuristic preview (linear)",
                line=dict(color="#7f8c8d", dash="dot", width=1.5),
            ),
            secondary_y=False,
        )

    _shade_meal_markers(fig, record.meals)
    _shade_exercise_spans(fig, record.exercise)

    fig.update_xaxes(title_text="Time (min)")
    fig.update_yaxes(title_text="Glucose (mg/dL)", secondary_y=False)
    fig.update_yaxes(title_text="Insulin (U/h)", secondary_y=True)
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=55, t=40, b=40),
        title_text="Twin-axis: glucose + insulin",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------


def _build_env(
    profile: PatientProfile,
    meals: list[dict],
    exercise: list[dict],
    seed: Optional[int],
) -> DiabetesPIDEnv:
    env = DiabetesPIDEnv(
        patient_params=profile.patient_params,
        patient_weight=profile.patient_weight,
        target_glucose=profile.target_glucose,
        seed=seed,
        observation_noise_std=profile.observation_noise_std,
        carb_ratio_override=profile.carb_ratio,
        isf_override=profile.isf,
        # Skip random scenario reload to honour the injected schedules.
        test_case_id=None,
    )
    env.set_meal_schedule(meals)
    env.set_exercise_schedule(exercise)
    env._skip_reload = True  # tell DiabetesPIDEnv.reset() to keep our schedule
    env.patient_weight = profile.patient_weight
    return env


def run_demo_episode(
    profile: PatientProfile,
    meals: list[dict],
    exercise: list[dict],
    controller: str,
    horizon: int,
    seed: Optional[int],
) -> Optional[EpisodeRecord]:
    """Run one episode. For ``controller=='rl'``, returns ``None`` if no
    actor weights are present so the UI does not mislabel a baseline run
    as RL.
    """
    env = _build_env(profile, meals, exercise, seed)
    actor = None
    if controller == "rl":
        actor = _load_actor_cached()
        if actor is None:
            return None
    record = run_episode(
        env,
        controller=controller,
        max_steps=horizon,
        actor=actor,
        seed=seed,
    )
    return record


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    set_global_seed(int(os.environ.get("AP_RL_SEED") or 0) or None)

    demo_defaults = _demo_defaults()
    target_band_cfg = demo_defaults.get("target_band", {})
    target_band = (
        float(target_band_cfg.get("low", 70.0)),
        float(target_band_cfg.get("high", 180.0)),
    )

    st.title("🩺 Artificial Pancreas — RL-tuned PID Digital Twin")
    st.caption(
        "Synthetic Type-1 Diabetes simulation. Research/demo only — **not** a "
        "medical device, **not** clinically validated."
    )

    with st.sidebar:
        st.header("Scenario")

        profile_names = _available_profiles()
        default_profile = demo_defaults.get("default_profile", profile_names[0])
        if default_profile not in profile_names:
            default_profile = profile_names[0]
        profile_name = st.selectbox(
            "Patient profile",
            profile_names,
            index=profile_names.index(default_profile),
            help="Synthetic patient (insulin sensitivity, ISF, noise).",
        )
        profile = load_profile(
            profile_name, base_patient_params=dict(DEFAULT_PATIENT_PARAMS)
        )
        with st.expander("Profile description"):
            st.markdown(profile.description or "_No description provided._")
            st.markdown(
                f"- Body weight: **{profile.patient_weight:.1f} kg**\n"
                f"- Target glucose: **{profile.target_glucose:.0f} mg/dL**\n"
                f"- Observation noise σ: **{profile.observation_noise_std:.1f} mg/dL**\n"
                f"- Carb ratio override: **{profile.carb_ratio}**\n"
                f"- ISF override: **{profile.isf}**"
            )

        templates = _meal_templates()
        default_template = demo_defaults.get(
            "default_meal_template", next(iter(templates), "three_meals_active")
        )
        if default_template not in templates:
            default_template = next(iter(templates), default_template)
        template_name = st.selectbox(
            "Meal template", list(templates) or [default_template]
        )
        meals_default, exercise_default = ([], [])
        if templates and template_name in templates:
            meals_default, exercise_default = build_scenario(templates[template_name])

        with st.expander("Meal schedule (editable)"):
            meal_df = pd.DataFrame(meals_default or [{"time": 420, "carbs": 45}])
            edited_meals = st.data_editor(
                meal_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "time": st.column_config.NumberColumn(
                        "Time (min)", min_value=0, max_value=1440, step=15
                    ),
                    "carbs": st.column_config.NumberColumn(
                        "Carbs (g)", min_value=0, max_value=200, step=5
                    ),
                },
            )
            meals_runtime = [
                {"time": float(r["time"]), "carbs": float(r["carbs"])}
                for _, r in edited_meals.iterrows()
                if r["carbs"] > 0
            ]

        exercise_runtime = exercise_default
        toggle_exercise = st.toggle(
            "Include scheduled exercise", value=bool(exercise_default)
        )
        if not toggle_exercise:
            exercise_runtime = []

        horizon = st.slider(
            "Simulation horizon (min)",
            min_value=120,
            max_value=int(demo_defaults.get("default_horizon_min", 1440)),
            value=int(demo_defaults.get("default_horizon_min", 1440)),
            step=60,
        )

        seed = st.number_input(
            "Random seed (optional)",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            help="Reproducible scenario picks + observation noise.",
        )

        st.header("Controllers")
        compare_mode = st.toggle("Compare RL vs Baseline PID", value=True)
        run_button = st.button("▶ Run rollout", type="primary", use_container_width=True)

    if not run_button:
        st.info(
            "Choose a profile, meal template, and horizon, then click **Run rollout**.\n"
            "If you have not downloaded checkpoints yet, the RL controller falls back "
            "to baseline PID automatically."
        )
        st.stop()

    progress = st.progress(0.0, text="Running baseline PID...")
    try:
        baseline_record = run_demo_episode(
            profile,
            meals_runtime,
            exercise_runtime,
            controller="baseline",
            horizon=horizon,
            seed=int(seed),
        )
    except Exception as exc:
        st.error(f"Baseline rollout failed: {exc}")
        st.stop()
    progress.progress(0.5, text="Running RL controller...")

    rl_record: Optional[EpisodeRecord] = None
    if compare_mode:
        try:
            rl_record = run_demo_episode(
                profile,
                meals_runtime,
                exercise_runtime,
                controller="rl",
                horizon=horizon,
                seed=int(seed),
            )
        except Exception as exc:
            st.error(f"RL rollout failed: {exc}")
            rl_record = None

    progress.progress(1.0, text="Rendering charts...")
    progress.empty()

    # RL status (avoid implying RL ran when weights are missing) ------------
    ckpt_dir = checkpoints_dir()
    actor_path = ckpt_dir / ACTOR_BEST
    if compare_mode:
        if rl_record is not None and rl_record.controller == "rl":
            st.success(
                f"RL controller active — loaded `{actor_path.name}` from `{ckpt_dir}`."
            )
        else:
            st.error(
                "**RL comparison unavailable.** No trained actor at "
                f"`{actor_path}` (or legacy `diabetes_actor_best.h5` beside it). "
                "Baseline PID below is still valid; RL curves are not.\n\n"
                "**Next steps:** (1) `ap-rl-train --preset default` until "
                "`diabetes_actor_best.weights.h5` appears in `checkpoints/`, **or** (2) set "
                "`AP_RL_CHECKPOINT_URL` to your **real** release URL (not the README "
                "placeholder) and run `ap-rl-download`. After adding files manually, "
                "use the Streamlit menu **Clear cache** then **Rerun** so the actor reloads."
            )

    # Headline metrics ---------------------------------------------------------
    primary_record = rl_record if (rl_record and rl_record.controller == "rl") else baseline_record
    stats = primary_record.stats or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean BGL", f"{stats.get('mean_glucose', 0):.1f} mg/dL")
    col2.metric(
        "TIR 70-180",
        f"{stats.get('time_in_range_70_180', 0):.1f}%",
    )
    col3.metric(
        "Time <70 (hypo)",
        f"{stats.get('time_hypo_70', 0):.2f}%",
        delta=f"-{stats.get('time_hypo_70', 0):.2f}",
        delta_color="inverse",
    )
    col4.metric(
        "Total insulin",
        f"{stats.get('total_insulin', 0):.1f} U",
    )

    # Playback slider ----------------------------------------------------------
    max_step = len(primary_record.times)
    playback_idx = st.slider(
        "Playback (minutes shown)",
        min_value=10,
        max_value=max_step,
        value=max_step,
        step=10,
    )

    show_preview = bool(demo_defaults.get("display", {}).get("show_heuristic_prediction", True))

    # Charts -------------------------------------------------------------------
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Digital twin — glucose")
        st.plotly_chart(
            build_glucose_figure(
                primary_record,
                target_band=target_band,
                show_preview=show_preview,
                horizon=playback_idx,
            ),
            use_container_width=True,
        )
        st.subheader("Insulin delivery")
        st.plotly_chart(
            build_insulin_figure(primary_record, horizon=playback_idx),
            use_container_width=True,
        )
        st.subheader("Twin-axis overlay (glucose + insulin)")
        st.caption("Shared time axis; insulin uses the right-hand scale.")
        st.plotly_chart(
            build_glucose_insulin_twin_figure(
                primary_record,
                target_band=target_band,
                show_preview=show_preview,
                horizon=playback_idx,
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("Glucose distribution")
        dist = {
            "Hypo <70": stats.get("time_hypo_70", 0.0),
            "TIR 70-180": stats.get("time_in_range_70_180", 0.0),
            "Hyper >180": stats.get("time_hyper_180", 0.0),
        }
        st.bar_chart(pd.DataFrame({"%": dist}))
        st.caption("Per-minute classification across the simulated day.")

        if compare_mode and rl_record is not None and rl_record.controller == "rl":
            st.subheader("Controller comparison")
            cmp_df = pd.DataFrame(
                {
                    "Time (min)": baseline_record.times[:playback_idx],
                    "Baseline PID": baseline_record.glucose[:playback_idx],
                    "RL-tuned": rl_record.glucose[:playback_idx],
                }
            ).set_index("Time (min)")
            st.line_chart(cmp_df)

    with st.expander("PID gain trajectory"):
        gains = pd.DataFrame(
            {
                "Time (min)": primary_record.times[:playback_idx],
                "Kp": primary_record.Kp[:playback_idx],
                "Ki": primary_record.Ki[:playback_idx],
                "Kd": primary_record.Kd[:playback_idx],
            }
        ).set_index("Time (min)")
        st.line_chart(gains)

    if rl_record is None and compare_mode:
        st.caption(
            "No RL trajectory to plot — see the red **RL comparison unavailable** banner above."
        )
    elif rl_record is None and not compare_mode:
        st.info(
            "Compare mode is off. Enable the toggle and install checkpoints to overlay "
            "an RL-tuned trajectory."
        )

    st.markdown(
        "---\n"
        "**Disclaimer.** This simulator is intended for research, education, and "
        "engineering-portfolio demonstrations only. The synthetic profiles do **not** "
        "represent real patients. Do not use the predictions, gain trajectories, or "
        "insulin recommendations shown here to inform real clinical decisions."
    )


if __name__ == "__main__":
    main()
