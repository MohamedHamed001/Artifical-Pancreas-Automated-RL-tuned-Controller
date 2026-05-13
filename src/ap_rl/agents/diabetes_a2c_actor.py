"""A2C actor network for PID-delta control.

Architecture preserved verbatim from the legacy
``RL_Diabetes_Controller/A2C/diabetes_a2c_actor.py``:
4 dense layers, dropout, ``tanh`` output scaled by ``action_bound``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
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
        state_input = Input(shape=(self.state_dim,))

        h1 = Dense(128, activation="relu")(state_input)
        h1 = tf.keras.layers.Dropout(0.2)(h1)
        h2 = Dense(128, activation="relu")(h1)
        h2 = tf.keras.layers.Dropout(0.2)(h2)
        h3 = Dense(64, activation="relu")(h2)
        h3 = tf.keras.layers.Dropout(0.1)(h3)
        h4 = Dense(32, activation="relu")(h3)

        delta_pid = Dense(self.action_dim, activation="tanh")(h4)
        scaled_output = Lambda(lambda x: x * self.action_bound)(delta_pid)

        return Model(inputs=state_input, outputs=scaled_output)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Sample an action (deterministic mean) from the current policy."""
        state = np.reshape(state, [1, self.state_dim])
        action = self.model(state)[0]
        return action.numpy()

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> float:
        """One policy-gradient step (Gaussian log-prob, fixed std=0.1)."""
        with tf.GradientTape() as tape:
            predicted_actions = self.model(states)
            std = 0.1
            log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - predicted_actions) / std), axis=1
            ) - 0.5 * self.action_dim * tf.math.log(2 * np.pi * std**2)
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            entropy = 0.5 * self.action_dim * (1 + tf.math.log(2 * np.pi * std**2))
            entropy_loss = -0.01 * entropy
            total_loss = policy_loss + entropy_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return float(total_loss.numpy())

    def save_weights(self, filepath: str) -> None:
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        self.model.load_weights(filepath)
