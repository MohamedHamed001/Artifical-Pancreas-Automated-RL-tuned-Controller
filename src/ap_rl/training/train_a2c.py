"""Consolidated A2C training entrypoint with named presets.

Replaces the four legacy trainers (``advanced_training.py``,
``robust_training.py``, ``fixed_diabetes_trainer.py``,
``simple_effective_trainer.py``) with a single ``--preset`` switch.

Presets:

* ``default``: original ``diabetes_a2c_main.py`` settings.
* ``robust``: longer early-stopping patience and higher initial noise.
* ``conservative``: smaller LR for stability-focused fine-tuning.

The buggy ``hovorka_gym_env.HovorkaPatient`` reference present in the
legacy ``fixed_diabetes_trainer.py`` is dropped - the consolidated
trainer always uses :class:`ap_rl.envs.DiabetesPIDEnv` directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from ap_rl.agents import DiabetesA2CAgent
from ap_rl.envs import DiabetesPIDEnv
from ap_rl.utils.seed import set_global_seed


DEFAULT_PATIENT_PARAMS = {
    "BW": 75,
    "k_a1": 0.006,
    "k_a2": 0.06,
    "k_a3": 0.05,
    "k_b1": 0.003,
    "k_b2": 0.06,
    "k_b3": 0.04,
    "k_c1": 0.5,
    "V_I": 0.12,
    "t_max_I": 55,
    "k_e": 0.138,
    "F_01": 0.0097,
    "V_G": 0.16,
    "k_12": 0.066,
    "EGP_0": 0.0161,
    "AG": 1.0,
    "t_max_G": 30,
    "G_init": 10.0,
    "A_EGP": 0.05,
    "phi_EGP": -60,
    "F_peak": 1.35,
    "K_rise": 5.0,
    "K_decay": 0.01,
    "G_thresh": 9.0,
    "k_R": 0.0031,
}


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
}


def build_env(seed: Optional[int]) -> DiabetesPIDEnv:
    """Construct the canonical training env with default Hovorka params."""
    return DiabetesPIDEnv(
        patient_params=dict(DEFAULT_PATIENT_PARAMS),
        patient_weight=75,
        target_glucose=120,
        seed=seed,
    )


def train(
    preset: str = "default",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> DiabetesA2CAgent:
    """Run training under the named preset and return the trained agent."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
    config = PRESETS[preset]

    set_global_seed(seed)

    env = build_env(seed=seed)
    agent = DiabetesA2CAgent(env)
    agent.actor_lr = config.actor_lr
    agent.critic_lr = config.critic_lr
    agent.exploration_noise = config.exploration_noise
    agent.early_stopping_patience = config.early_stopping_patience

    if verbose:
        print(f"Training with preset '{config.name}'")
        print(
            f"  max_episodes={config.max_episodes}, "
            f"actor_lr={config.actor_lr}, critic_lr={config.critic_lr}, "
            f"noise={config.exploration_noise}, "
            f"patience={config.early_stopping_patience}"
        )

    agent.train(
        max_episodes=config.max_episodes,
        plot_progress=False,
        save_frequency=config.save_frequency,
        verbose=verbose,
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
