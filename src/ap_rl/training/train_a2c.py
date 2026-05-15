"""Consolidated A2C training entrypoint with named presets.

Replaces the four legacy trainers (``advanced_training.py``,
``robust_training.py``, ``fixed_diabetes_trainer.py``,
``simple_effective_trainer.py``) with a single ``--preset`` switch.

Presets:

* ``default``: original ``diabetes_a2c_main.py`` settings.
* ``robust``: longer early-stopping patience and higher initial noise.
* ``conservative``: smaller LR for stability-focused fine-tuning.
* ``generalize``: anti-overfitting preset — evaluates on held-out scenarios
  every 25 episodes and saves the checkpoint with the best *mean TIR across
  those held-out cases* (not the best single-episode reward).  This is the
  recommended preset when TIR variance across scenarios is the main concern.

The buggy ``hovorka_gym_env.HovorkaPatient`` reference present in the
legacy ``fixed_diabetes_trainer.py`` is dropped - the consolidated
trainer always uses :class:`ap_rl.envs.DiabetesPIDEnv` directly.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ap_rl.envs import DiabetesPIDEnv
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.envs.profile_loader import load_profile, list_profiles
from ap_rl.utils.paths import data_dir
from ap_rl.utils.seed import set_global_seed


@dataclass
class PresetConfig:
    """Training preset hyperparameter bundle."""

    name: str
    max_episodes: int
    actor_lr: float
    critic_lr: float
    exploration_noise: float
    early_stopping_patience: int
    save_frequency: int
    # Held-out evaluation settings (only used by the 'generalize' preset)
    eval_frequency: int = 0          # 0 = disabled, N = evaluate every N episodes
    num_eval_cases: int = 20         # how many held-out scenarios to evaluate on
    eval_case_seed: int = 99         # seed for selecting held-out case IDs
    noise_min: float = 0.05          # exploration noise floor


PRESETS: dict[str, PresetConfig] = {
    "default": PresetConfig(
        name="default",
        max_episodes=500,
        actor_lr=0.0005,
        critic_lr=0.001,
        exploration_noise=0.15,
        early_stopping_patience=150,
        save_frequency=25,
    ),
    "robust": PresetConfig(
        name="robust",
        max_episodes=500,
        actor_lr=0.0005,
        critic_lr=0.001,
        exploration_noise=0.20,
        early_stopping_patience=200,
        save_frequency=25,
    ),
    "conservative": PresetConfig(
        name="conservative",
        max_episodes=300,
        actor_lr=0.0001,
        critic_lr=0.0005,
        exploration_noise=0.10,
        early_stopping_patience=300,
        save_frequency=50,
    ),
    "generalize": PresetConfig(
        name="generalize",
        max_episodes=400,
        actor_lr=0.0003,
        critic_lr=0.0008,
        exploration_noise=0.18,
        early_stopping_patience=150,
        save_frequency=40,
        # Held-out eval: evaluate on 12 scenarios every 40 episodes
        # and save the checkpoint with the best cross-scenario mean TIR.
        eval_frequency=40,
        num_eval_cases=12,
        eval_case_seed=99,
        noise_min=0.05,
    ),
    "allprofiles": PresetConfig(
        name="allprofiles",
        max_episodes=600,
        actor_lr=0.0003,
        critic_lr=0.0006,
        exploration_noise=0.20,
        early_stopping_patience=200,
        save_frequency=50,
        # Multi-profile held-out eval: all profiles, all scenario IDs
        eval_frequency=50,
        num_eval_cases=16,
        eval_case_seed=42,
        noise_min=0.04,
    ),
}


# ---------------------------------------------------------------------------
# Physiological parameter ranges for random patient sampling
# (used by the allprofiles preset curriculum)
# ---------------------------------------------------------------------------
_PROFILE_RANGES = {
    # ISF range (mg/dL per U): 20 (resistant) to 80 (very sensitive)
    "isf": (20.0, 80.0),
    # Carb ratio range (g per U): 6 (resistant) to 20 (sensitive)
    "carb_ratio": (6.0, 20.0),
    # Body weight range (kg)
    "weight": (55.0, 100.0),
    # k_e (insulin elimination, 1/min): 0.10 to 0.18
    "k_e": (0.10, 0.18),
    # EGP_0 baseline (mmol/min/kg): 0.012 to 0.020
    "EGP_0": (0.012, 0.020),
}

_PROFILE_NAMES: list[str] = []   # filled lazily on first use


def _sample_random_patient(rng: random.Random) -> dict:
    """Sample a random physiologically plausible patient configuration.

    Returns a kwargs dict suitable for passing directly to DiabetesPIDEnv.
    This is the core mechanism that forces the agent to learn a universal
    policy that works across all patient types.
    """
    weight = rng.uniform(*_PROFILE_RANGES["weight"])
    isf    = rng.uniform(*_PROFILE_RANGES["isf"])
    cr     = rng.uniform(*_PROFILE_RANGES["carb_ratio"])
    k_e    = rng.uniform(*_PROFILE_RANGES["k_e"])
    EGP_0  = rng.uniform(*_PROFILE_RANGES["EGP_0"])

    params = dict(DEFAULT_PATIENT_PARAMS)
    params["BW"]   = weight
    params["k_e"]  = k_e
    params["EGP_0"] = EGP_0

    return {
        "patient_params": params,
        "patient_weight": weight,
        "isf_override": isf,
        "carb_ratio_override": cr,
    }


def build_env(
    seed: Optional[int],
    case_id: Optional[int] = None,
    patient_kwargs: Optional[dict] = None,
) -> DiabetesPIDEnv:
    """Construct a training env.

    Args:
        seed: RNG seed for scenario selection.
        case_id: If set, pin the env to this specific scenario.
        patient_kwargs: Optional dict with ``patient_params``,
            ``patient_weight``, ``isf_override``, ``carb_ratio_override``.
            When None, uses the default Hovorka 75 kg patient.
    """
    kwargs = patient_kwargs or {
        "patient_params": dict(DEFAULT_PATIENT_PARAMS),
        "patient_weight": 75,
        "isf_override": None,
        "carb_ratio_override": None,
    }
    return DiabetesPIDEnv(
        patient_params=kwargs["patient_params"],
        patient_weight=kwargs["patient_weight"],
        target_glucose=120,
        seed=seed,
        test_case_id=case_id,
        isf_override=kwargs.get("isf_override"),
        carb_ratio_override=kwargs.get("carb_ratio_override"),
    )


def _select_held_out_cases(
    all_cases: list[int],
    n: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split scenario IDs into train / held-out sets.

    Returns:
        (train_cases, held_out_cases) — held-out cases are excluded from
        the training env's random scenario pool during generalize preset.
        (Currently the env picks randomly from all files anyway; this
        function returns the IDs used for the periodic evaluation only.)
    """
    rng = random.Random(seed)
    shuffled = list(all_cases)
    rng.shuffle(shuffled)
    held_out = shuffled[:n]
    train = shuffled[n:]
    return train, held_out


def _get_all_case_ids() -> list[int]:
    """Return sorted list of all available scenario IDs from data dir."""
    import glob
    import os

    pattern = os.path.join(os.fspath(data_dir()), "MealData_case*.data")
    ids: list[int] = []
    for p in glob.glob(pattern):
        name = os.path.basename(p)
        try:
            ids.append(int(name.replace("MealData_case", "").replace(".data", "")))
        except ValueError:
            pass
    return sorted(ids)


def train(
    preset: str = "default",
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """Run training under the named preset and return the trained agent.

    TensorFlow is imported lazily here so importing the module does not
    require ``tensorflow`` to be installed (useful for the demo and the
    baseline smoke script).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
    config = PRESETS[preset]

    set_global_seed(seed)

    from ap_rl.agents import DiabetesA2CAgent  # noqa: PLC0415 - lazy TF import
    from ap_rl.utils.hardware import configure_for_training  # noqa: PLC0415
    configure_for_training(verbose=verbose)

    is_allprofiles = (preset == "allprofiles")
    patient_rng = random.Random(seed if seed is not None else 7)

    if is_allprofiles:
        # Start curriculum: sample a mid-range patient for the first episode.
        init_patient = {
            "patient_params": dict(DEFAULT_PATIENT_PARAMS),
            "patient_weight": 75,
            "isf_override": 36.0,
            "carb_ratio_override": 12.0,
        }
    else:
        init_patient = None

    env = build_env(seed=seed, patient_kwargs=init_patient)
    agent = DiabetesA2CAgent(env)
    agent.actor_lr = config.actor_lr
    agent.critic_lr = config.critic_lr
    agent.exploration_noise = config.exploration_noise
    agent.early_stopping_patience = config.early_stopping_patience
    agent.noise_min = config.noise_min

    # Held-out evaluation setup
    eval_case_ids: Optional[list[int]] = None
    eval_env_factory = None
    if config.eval_frequency > 0:
        all_ids = _get_all_case_ids()
        if len(all_ids) < config.num_eval_cases:
            print(
                f"Warning: only {len(all_ids)} scenarios found, "
                f"using all for held-out eval."
            )
            eval_case_ids = all_ids
        else:
            _, eval_case_ids = _select_held_out_cases(
                all_ids, config.num_eval_cases, config.eval_case_seed
            )

        if is_allprofiles:
            # Evaluate on ALL 4 profiles × held-out cases to get a true
            # cross-profile TIR score.
            _isf_test_values = [36.0, 60.0, 80.0, 20.0]  # standard/sensitive/very-sensitive/resistant
            _cr_test_values  = [12.0, 14.0, 10.0,  8.0]
            _w_test_values   = [75.0, 65.0, 70.0, 90.0]

            def eval_env_factory(cid: int) -> DiabetesPIDEnv:
                # Round-robin through 4 patient archetypes
                idx = cid % 4
                p = dict(DEFAULT_PATIENT_PARAMS)
                p["BW"] = _w_test_values[idx]
                return build_env(
                    seed=0,
                    case_id=cid,
                    patient_kwargs={
                        "patient_params": p,
                        "patient_weight": _w_test_values[idx],
                        "isf_override": _isf_test_values[idx],
                        "carb_ratio_override": _cr_test_values[idx],
                    },
                )
        else:
            eval_env_factory = lambda cid: build_env(seed=0, case_id=cid)  # noqa: E731

    if verbose:
        print(f"Training with preset '{config.name}'")
        print(
            f"  max_episodes={config.max_episodes}, "
            f"actor_lr={config.actor_lr}, critic_lr={config.critic_lr}, "
            f"noise={config.exploration_noise} (min={config.noise_min}), "
            f"patience={config.early_stopping_patience}"
        )
        if is_allprofiles:
            print("  Multi-profile curriculum: random patient sampled each episode")
            print("  Patient ranges: ISF 20–80, CR 6–20, Weight 55–100 kg")
        if eval_case_ids:
            print(
                f"  Held-out eval: {len(eval_case_ids)} scenarios "
                f"every {config.eval_frequency} episodes "
                f"(IDs: {eval_case_ids[:5]}{'...' if len(eval_case_ids) > 5 else ''})"
            )
            print("  Best checkpoint = highest mean TIR across held-out scenarios")
        else:
            print("  Best checkpoint = best rolling-100-episode avg reward")

    # -----------------------------------------------------------------------
    # Multi-profile curriculum wrapper
    # -----------------------------------------------------------------------
    # For the allprofiles preset we rebuild the env at the start of each
    # episode with a freshly sampled patient.  This is implemented by
    # monkey-patching the agent's env reference inside the training loop.
    # The agent itself calls self.env.reset() at each episode start, so
    # swapping the env reference here is sufficient.
    original_train = agent.train

    if is_allprofiles:
        def _multi_profile_train(
            max_episodes,
            plot_progress=False,
            save_frequency=50,
            verbose=True,
            eval_case_ids=None,
            eval_env_factory=None,
            eval_frequency=25,
        ):
            """Wraps the standard training loop, swapping patient each episode."""
            use_tir_criterion = (eval_case_ids is not None and eval_env_factory is not None)
            best_eval_tir = -float("inf")

            from collections import deque
            import numpy as np
            recent_rewards: deque = deque(maxlen=100)

            for episode in range(max_episodes):
                # Sample a new random patient for this episode
                new_patient = _sample_random_patient(patient_rng)
                agent.env = build_env(seed=None, patient_kwargs=new_patient)

                traj_states:      list = []
                traj_actions:     list = []
                traj_next_states: list = []
                traj_rewards:     list = []
                traj_dones:       list = []

                episode_reward = 0.0
                episode_length = 0
                state = agent.env.reset()
                done = False

                if verbose:
                    isf = agent.env.insulin_calc.isf
                    cr  = agent.env.insulin_calc.carb_ratio
                    bw  = agent.env.patient_weight
                    print(
                        f"\n=== Episode {episode + 1}/{max_episodes} === "
                        f"[ISF={isf:.0f}, CR={cr:.1f}, BW={bw:.0f}kg]",
                        flush=True,
                    )

                while not done:
                    action = agent.actor.get_action(state)
                    noise  = np.random.normal(0, agent.exploration_noise, size=action.shape)
                    action = np.clip(action + noise, -agent.action_bound, agent.action_bound)

                    next_state, reward, done, info = agent.env.step(action)

                    traj_states.append(state)
                    traj_actions.append(action)
                    traj_next_states.append(next_state)
                    traj_rewards.append(reward)
                    traj_dones.append(done)

                    episode_reward += reward
                    episode_length += 1

                    if verbose and episode_length % 200 == 0:
                        print(
                            f"  Step {episode_length}/1440: BGL={info['glucose']:.1f}, "
                            f"Reward={reward:.2f}, noise={agent.exploration_noise:.3f}",
                            flush=True,
                        )

                    if len(traj_states) >= agent.batch_size or done:
                        s  = np.array(traj_states,      dtype=np.float32)
                        a  = np.array(traj_actions,      dtype=np.float32)
                        ns = np.array(traj_next_states,  dtype=np.float32)
                        r  = np.array(traj_rewards,      dtype=np.float32).reshape(-1, 1)
                        d  = np.array(traj_dones,        dtype=bool)

                        import tensorflow as tf
                        all_states = np.vstack([s, ns]).astype(np.float32)
                        all_vals   = agent.critic.model(all_states, training=False).numpy()
                        v_vals     = all_vals[:len(s)]
                        nv_vals    = all_vals[len(s):]

                        td_targets = np.where(d.reshape(-1, 1), r, r + agent.gamma * nv_vals)
                        advantages = td_targets - v_vals
                        advantages = agent._normalize_advantages(advantages)

                        agent.critic.train_on_batch(s, td_targets)
                        agent.actor.train(s, a, advantages)

                        traj_states.clear(); traj_actions.clear()
                        traj_next_states.clear(); traj_rewards.clear()
                        traj_dones.clear()

                    state = next_state

                agent.exploration_noise = max(
                    agent.exploration_noise * agent.noise_decay, agent.noise_min
                )

                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                recent_rewards.append(episode_reward)

                stats = agent.env.get_statistics()
                agent.glucose_stats.append(stats)

                avg_reward = float(np.mean(recent_rewards))
                agent.running_avg_reward.append(avg_reward)

                if verbose:
                    print(
                        f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                        f"len={episode_length}, "
                        f"TIR70-180={stats.get('time_in_range_70_180', 0):.1f}%, "
                        f"noise={agent.exploration_noise:.4f}, "
                        f"recent100_avg={avg_reward:.2f}"
                    )

                if use_tir_criterion and (episode + 1) % eval_frequency == 0:
                    eval_result = agent.eval_on_scenarios(eval_case_ids, eval_env_factory)
                    mean_tir = eval_result["mean_tir"]
                    if verbose:
                        print(
                            f"  [Eval] held-out mean TIR={mean_tir:.1f}% "
                            f"± {eval_result['std_tir']:.1f}%  "
                            f"min={eval_result['min_tir']:.1f}%"
                        )
                    if mean_tir > best_eval_tir:
                        best_eval_tir = mean_tir
                        agent.save_weights("best")
                        if verbose:
                            print(f"  [Eval] ✓ New best checkpoint (TIR={mean_tir:.1f}%)")

                elif not use_tir_criterion:
                    if avg_reward > agent.best_avg_reward:
                        agent.best_avg_reward = avg_reward
                        agent.save_weights("best")

                if (episode + 1) % save_frequency == 0:
                    agent.save_weights(f"episode_{episode + 1}")

            agent.save_weights("final")

        agent.train = _multi_profile_train

    agent.train(
        max_episodes=config.max_episodes,
        plot_progress=False,
        save_frequency=config.save_frequency,
        verbose=verbose,
        eval_case_ids=eval_case_ids,
        eval_env_factory=eval_env_factory,
        eval_frequency=config.eval_frequency if config.eval_frequency > 0 else 25,
    )
    return agent


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the A2C PID-tuning agent on the Hovorka virtual patient.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="default",
        help="Hyperparameter preset (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step logging.",
    )
    args = parser.parse_args(argv)
    train(preset=args.preset, seed=args.seed, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
