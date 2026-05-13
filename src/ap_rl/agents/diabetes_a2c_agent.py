"""A2C agent orchestrating the actor and critic for PID-delta control.

Hyperparameters, exploration schedule, and training loop preserved
verbatim from the legacy implementation. Only logging and weight paths
were touched - we now resolve ``checkpoints/`` via
:func:`ap_rl.utils.paths.checkpoints_dir` instead of an in-tree
``save_weights/`` folder, and accept an explicit override.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np

from ap_rl.agents.diabetes_a2c_actor import DiabetesActor
from ap_rl.agents.diabetes_a2c_critic import DiabetesCritic
from ap_rl.utils.paths import checkpoints_dir


class DiabetesA2CAgent:
    """Advantage-Actor-Critic agent tuning the PID gains in real time."""

    def __init__(self, env, save_dir: Optional[str | os.PathLike] = None) -> None:
        self.env = env

        _ = env.reset()
        self.state_dim = len(env._get_state())
        self.action_dim = 3
        self.action_bound = 0.1

        self.actor_lr = 0.0005
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.batch_size = 32

        self.actor = DiabetesActor(
            self.state_dim, self.action_dim, self.action_bound, self.actor_lr
        )
        self.critic = DiabetesCritic(self.state_dim, self.critic_lr)

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.glucose_stats: list[dict] = []
        self.running_avg_reward: list[float] = []
        self.best_avg_reward = -float("inf")
        self.episodes_without_improvement = 0
        self.early_stopping_patience = 150

        self.exploration_noise = 0.15
        self.noise_decay = 0.995

        self.save_dir = os.fspath(save_dir) if save_dir is not None else os.fspath(
            checkpoints_dir()
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def calculate_advantage_and_target(
        self, reward, v_value, next_v_value, done
    ):
        if done:
            td_target = reward
            advantage = td_target - v_value
        else:
            td_target = reward + self.gamma * next_v_value
            advantage = td_target - v_value
        return advantage, td_target

    def unpack_batch(self, batch):
        if len(batch) == 0:
            return np.array([])
        return np.vstack(batch)

    def train(
        self,
        max_episodes: int,
        plot_progress: bool = False,
        save_frequency: int = 50,
        verbose: bool = True,
    ) -> None:
        """Run A2C training for ``max_episodes`` episodes."""
        best_reward = -float("inf")
        recent_rewards: deque = deque(maxlen=100)

        for episode in range(max_episodes):
            batch_states: list = []
            batch_actions: list = []
            batch_td_targets: list = []
            batch_advantages: list = []
            episode_reward = 0.0
            episode_length = 0

            state = self.env.reset()
            done = False

            if verbose:
                print(f"\n=== Episode {episode + 1}/{max_episodes} ===")

            while not done:
                action = self.actor.get_action(state)
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)

                next_state, reward, done, info = self.env.step(action)

                state_batch = np.reshape(state, [1, self.state_dim])
                next_state_batch = np.reshape(next_state, [1, self.state_dim])
                action_batch = np.reshape(action, [1, self.action_dim])
                reward_batch = np.reshape(reward, [1, 1])

                v_value = self.critic.model(state_batch)
                next_v_value = self.critic.model(next_state_batch)

                advantage, td_target = self.calculate_advantage_and_target(
                    reward_batch, v_value, next_v_value, done
                )

                batch_states.append(state_batch)
                batch_actions.append(action_batch)
                batch_td_targets.append(td_target)
                batch_advantages.append(advantage)

                episode_reward += reward
                episode_length += 1

                if verbose and episode_length % 100 == 0:
                    print(
                        f"  Step {episode_length}: BGL={info['glucose']:.1f}, "
                        f"Basal={info['basal_insulin']:.2f}, "
                        f"Bolus={info['bolus_insulin']:.2f}, "
                        f"Total={info['total_insulin']:.2f}, "
                        f"PID=[{info['Kp']:.3f}, {info['Ki']:.3f}, "
                        f"{info['Kd']:.3f}], Reward={reward:.2f}"
                    )

                if len(batch_states) >= self.batch_size or done:
                    if len(batch_states) > 0:
                        states = self.unpack_batch(batch_states)
                        actions = self.unpack_batch(batch_actions)
                        td_targets = self.unpack_batch(batch_td_targets)
                        advantages = self.unpack_batch(batch_advantages)

                        self.critic.train_on_batch(states, td_targets)
                        self.actor.train(states, actions, advantages)

                        batch_states.clear()
                        batch_actions.clear()
                        batch_td_targets.clear()
                        batch_advantages.clear()

                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            recent_rewards.append(episode_reward)

            stats = self.env.get_statistics()
            self.glucose_stats.append(stats)

            if verbose:
                print(
                    f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                    f"len={episode_length}, "
                    f"TIR80-140={stats.get('time_in_range_80_140', 0):.1f}%, "
                    f"recent100={np.mean(recent_rewards):.2f}"
                )

            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_weights("best")

            if (episode + 1) % save_frequency == 0:
                self.save_weights(f"episode_{episode + 1}")

            avg_reward = float(np.mean(recent_rewards))
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1

            if self.episodes_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {episode + 1} episodes")
                break

        self.save_weights("final")

    def test(self, num_episodes: int = 1, render: bool = True, load_best: bool = True):
        """Test the trained policy and return (rewards, stats) lists."""
        if load_best:
            self.load_weights("best")

        test_rewards: list[float] = []
        test_stats: list[dict] = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.actor.get_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if render:
                    self.env.render()
            test_rewards.append(episode_reward)
            test_stats.append(self.env.get_statistics())

        return test_rewards, test_stats

    def save_weights(self, name: str) -> None:
        actor_path = os.path.join(self.save_dir, f"diabetes_actor_{name}.h5")
        critic_path = os.path.join(self.save_dir, f"diabetes_critic_{name}.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, name: str) -> bool:
        """Load actor + critic weights. Returns False when missing."""
        actor_path = os.path.join(self.save_dir, f"diabetes_actor_{name}.h5")
        critic_path = os.path.join(self.save_dir, f"diabetes_critic_{name}.h5")
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            return True
        return False
