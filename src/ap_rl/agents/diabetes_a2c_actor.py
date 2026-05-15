"""A2C actor network for PID-delta control.

Architecture upgrade (v2):
- Input layer widened to 256 units to handle the richer 19-D observation
  (patient-identity features ISF, carb_ratio, weight added in v2).
- Dropout replaced with LayerNormalization.  Dropout randomly zeroes
  activations which causes inconsistent gradient magnitudes across the wide
  patient-parameter range.  LayerNorm normalises activations per-batch,
  giving stable updates regardless of whether the agent is treating an
  insulin-sensitive or insulin-resistant patient.
- tanh output scaled by action_bound (unchanged).

Do NOT add @tf.function to get_action — TF CPU graph compilation freezes
silently on macOS M-series with this architecture.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DiabetesActor:
    """Continuous-action actor producing ``(dKp, dKi, dKd)``."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        learning_rate: float,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate

        self.model = self._build_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_network(self) -> Model:
        """256→128→64→32 with LayerNorm for cross-patient stability."""
        state_input = Input(shape=(self.state_dim,))

        # First layer wider: handles richer 19-D obs including patient identity
        h1 = Dense(256, activation="relu")(state_input)
        h1 = LayerNormalization()(h1)

        h2 = Dense(128, activation="relu")(h1)
        h2 = LayerNormalization()(h2)

        h3 = Dense(64, activation="relu")(h2)
        h3 = LayerNormalization()(h3)

        h4 = Dense(32, activation="relu")(h3)

        delta_pid = Dense(self.action_dim, activation="tanh")(h4)
        scaled_output = Lambda(lambda x: x * self.action_bound)(delta_pid)

        return Model(inputs=state_input, outputs=scaled_output)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Return the deterministic action from the current policy.

        ``training=False`` is required so that Dropout layers are disabled
        during inference.  Calling ``self.model(state)`` without this flag
        leaves Dropout active (training mode default in TF), which randomly
        zeroes activations and produces noisy, degraded PID-delta actions.
        """
        state = np.reshape(state, [1, self.state_dim])
        action = self.model(state, training=False)[0]
        return action.numpy()

    def _train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        advantages: tf.Tensor,
    ) -> tf.Tensor:
        """Eager inner training step.

        @tf.function was removed: first-trace freezes the process for
        several minutes in the real training context (known TF issue with
        GradientTape + Dropout inside tf.function on CPU).  For this tiny
        128→64→32 network eager execution is fast enough — network updates
        are <3% of total episode wall-time.
        """
        std = tf.constant(0.1, dtype=tf.float32)
        two_pi_std2 = tf.cast(2.0 * np.pi * 0.1**2, tf.float32)

        with tf.GradientTape() as tape:
            predicted_actions = self.model(states, training=True)

            # Policy gradient loss (Gaussian log-prob weighted by advantage)
            log_probs = (
                -0.5 * tf.reduce_sum(tf.square((actions - predicted_actions) / std), axis=1)
                - 0.5 * tf.cast(self.action_dim, tf.float32) * tf.math.log(two_pi_std2)
            )
            policy_loss = -tf.reduce_mean(log_probs * tf.squeeze(advantages, axis=1))

            # Adaptive entropy bonus scaled by mean |advantage|
            mean_abs_adv = tf.reduce_mean(tf.abs(advantages)) + 1e-8
            entropy_per_dim = 0.5 * (1.0 + tf.math.log(two_pi_std2))
            entropy_bonus = 0.01 * entropy_per_dim * tf.cast(self.action_dim, tf.float32) * mean_abs_adv

            total_loss = policy_loss - entropy_bonus

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return total_loss

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> float:
        """One policy-gradient step with per-sample Gaussian entropy bonus.

        Delegates to :meth:`_train_step` which is JIT-compiled via
        ``@tf.function`` for maximum CPU throughput.
        """
        loss = self._train_step(
            tf.constant(states, dtype=tf.float32),
            tf.constant(actions, dtype=tf.float32),
            tf.constant(advantages, dtype=tf.float32),
        )
        return float(loss.numpy())

    def save_weights(self, filepath: str) -> None:
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        self.model.load_weights(filepath)
