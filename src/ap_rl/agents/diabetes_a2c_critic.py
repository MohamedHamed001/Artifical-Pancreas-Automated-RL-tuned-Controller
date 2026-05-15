"""A2C critic network estimating the state value V(s)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DiabetesCritic:
    """Single-scalar state-value head used by the A2C agent."""

    def __init__(self, state_dim: int, learning_rate: float) -> None:
        self.state_dim = state_dim
        self.learning_rate = learning_rate

        self.model = self._build_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_network(self) -> Model:
        state_input = Input(shape=(self.state_dim,))
        h1 = Dense(64, activation="relu")(state_input)
        h2 = Dense(64, activation="relu")(h1)
        h3 = Dense(32, activation="relu")(h2)
        value_output = Dense(1, activation="linear")(h3)
        return Model(inputs=state_input, outputs=value_output)

    def get_value(self, state: np.ndarray) -> float:
        state = np.reshape(state, [1, self.state_dim])
        value = self.model(state)[0]
        return float(value.numpy()[0])

    def train_on_batch(self, states: np.ndarray, td_targets: np.ndarray) -> float:
        """Eager MSE update. @tf.function removed — first-trace caused multi-minute hang."""
        s = tf.constant(states, dtype=tf.float32)
        t = tf.constant(td_targets, dtype=tf.float32)
        with tf.GradientTape() as tape:
            predicted_values = self.model(s, training=True)
            loss = tf.reduce_mean(tf.square(t - predicted_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return float(loss.numpy())

    def save_weights(self, filepath: str) -> None:
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        self.model.load_weights(filepath)
