import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class DiabetesCritic:
    """
    Critic network for diabetes PID parameter tuning using A2C.
    Estimates the value function for given states.
    """
    
    def __init__(self, state_dim, learning_rate):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of state space
            learning_rate (float): Learning rate for the optimizer
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        
        # Build the neural network
        self.model = self._build_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
    def _build_network(self):
        """Build the critic neural network."""
        # Input layer
        state_input = Input(shape=(self.state_dim,))
        
        # Hidden layers
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(64, activation='relu')(h1)
        h3 = Dense(32, activation='relu')(h2)
        
        # Output layer - single value (state value)
        value_output = Dense(1, activation='linear')(h3)
        
        model = Model(inputs=state_input, outputs=value_output)
        model.summary()
        
        return model
    
    def get_value(self, state):
        """
        Get state value from the current state.
        
        Args:
            state (np.array): Current state
            
        Returns:
            float: State value
        """
        state = np.reshape(state, [1, self.state_dim])
        value = self.model(state)[0]
        return value.numpy()[0]
    
    def train_on_batch(self, states, td_targets):
        """
        Train the critic network on a batch of data.
        
        Args:
            states (np.array): Batch of states
            td_targets (np.array): Batch of TD targets
        """
        with tf.GradientTape() as tape:
            # Get predicted values
            predicted_values = self.model(states)
            
            # Calculate MSE loss
            loss = tf.reduce_mean(tf.square(td_targets - predicted_values))
        
        # Calculate gradients and apply
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save_weights(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath) 