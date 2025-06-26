import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam

class DiabetesActor:
    """
    Actor network for diabetes PID parameter tuning using A2C.
    Outputs changes to PID parameters (delta_Kp, delta_Ki, delta_Kd).
    """
    
    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space (3 for PID parameters)
            action_bound (float): Maximum change per parameter per step
            learning_rate (float): Learning rate for the optimizer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        
        # Build the neural network
        self.model = self._build_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
    def _build_network(self):
        """Build the actor neural network."""
        # Input layer
        state_input = Input(shape=(self.state_dim,))
        
        # Enhanced hidden layers with dropout for better generalization
        h1 = Dense(128, activation='relu')(state_input)
        h1_dropout = tf.keras.layers.Dropout(0.2)(h1)
        
        h2 = Dense(128, activation='relu')(h1_dropout)
        h2_dropout = tf.keras.layers.Dropout(0.2)(h2)
        
        h3 = Dense(64, activation='relu')(h2_dropout)
        h3_dropout = tf.keras.layers.Dropout(0.1)(h3)
        
        h4 = Dense(32, activation='relu')(h3_dropout)
        
        # Output layer for PID parameter changes
        # Use tanh activation and scale by action_bound
        delta_pid = Dense(self.action_dim, activation='tanh')(h4)
        scaled_output = Lambda(lambda x: x * self.action_bound)(delta_pid)
        
        model = Model(inputs=state_input, outputs=scaled_output)
        model.summary()
        
        return model
    
    def get_action(self, state):
        """
        Get action (PID parameter changes) from the current state.
        
        Args:
            state (np.array): Current state
            
        Returns:
            np.array: Action (delta_Kp, delta_Ki, delta_Kd)
        """
        state = np.reshape(state, [1, self.state_dim])
        action = self.model(state)[0]
        return action.numpy()
    
    def train(self, states, actions, advantages):
        """
        Train the actor network using policy gradient.
        
        Args:
            states (np.array): Batch of states
            actions (np.array): Batch of actions taken
            advantages (np.array): Batch of advantages
        """
        with tf.GradientTape() as tape:
            # Get predicted actions
            predicted_actions = self.model(states)
            
            # Calculate action probabilities using Gaussian distribution
            # For continuous control, we use the predicted action as mean
            # and a fixed standard deviation
            std = 0.1  # Fixed standard deviation for exploration
            
            # Calculate log probabilities
            log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - predicted_actions) / std), axis=1
            ) - 0.5 * self.action_dim * tf.math.log(2 * np.pi * std**2)
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            
            # Add entropy bonus for exploration
            entropy = 0.5 * self.action_dim * (1 + tf.math.log(2 * np.pi * std**2))
            entropy_loss = -0.01 * entropy  # Small entropy coefficient
            
            total_loss = policy_loss + entropy_loss
        
        # Calculate gradients and apply
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss.numpy()
    
    def save_weights(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath) 