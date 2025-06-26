import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

from diabetes_a2c_actor import DiabetesActor
from diabetes_a2c_critic import DiabetesCritic

class DiabetesA2CAgent:
    """
    A2C Agent for learning to tune PID parameters for diabetes control.
    """
    
    def __init__(self, env):
        """
        Initialize the A2C agent.
        
        Args:
            env: Diabetes PID environment
        """
        self.env = env
        
        # Reset environment to ensure patient is initialized for state dimension calculation
        _ = env.reset()
        self.state_dim = len(env._get_state())
        self.action_dim = 3  # [delta_Kp, delta_Ki, delta_Kd]
        self.action_bound = 0.1  # Smaller action bound for stability
        
        # Improved hyperparameters for better stability
        self.actor_lr = 0.0005  # Reduced learning rate
        self.critic_lr = 0.001  # Reduced learning rate
        self.gamma = 0.99
        self.batch_size = 32  # Larger batch size
        
        # Create Actor and Critic networks
        self.actor = DiabetesActor(self.state_dim, self.action_dim, self.action_bound, self.actor_lr)
        self.critic = DiabetesCritic(self.state_dim, self.critic_lr)
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.glucose_stats = []
        self.running_avg_reward = []
        self.best_avg_reward = -float('inf')
        self.episodes_without_improvement = 0
        self.early_stopping_patience = 150  # Increased from 50 to 150
        
        # Enhanced exploration
        self.exploration_noise = 0.15  # Reduced noise as training progresses
        self.noise_decay = 0.995
        
        # Create save directory
        self.save_dir = os.path.join(os.path.dirname(__file__), 'save_weights')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def calculate_advantage_and_target(self, reward, v_value, next_v_value, done):
        """Calculate advantage and TD target."""
        if done:
            td_target = reward
            advantage = td_target - v_value
        else:
            td_target = reward + self.gamma * next_v_value
            advantage = td_target - v_value
        return advantage, td_target
    
    def unpack_batch(self, batch):
        """Extract data from batch."""
        if len(batch) == 0:
            return np.array([])
        return np.vstack(batch)
    
    def train(self, max_episodes, plot_progress=True, save_frequency=50, verbose=True):
        """
        Train the agent.
        
        Args:
            max_episodes (int): Maximum number of episodes to train
            plot_progress (bool): Whether to plot training progress
            save_frequency (int): How often to save weights
            verbose (bool): Whether to print progress
        """
        print(f"Starting diabetes control training for {max_episodes} episodes...")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Action bound: {self.action_bound}")
        
        best_reward = -float('inf')
        recent_rewards = deque(maxlen=100)  # Track recent performance
        
        for episode in range(max_episodes):
            # Initialize episode
            batch_states, batch_actions, batch_td_targets, batch_advantages = [], [], [], []
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            state = self.env.reset()
            done = False
            
            print(f"\n=== Episode {episode + 1}/{max_episodes} ===")
            
            while not done:
                # Get action from actor
                action = self.actor.get_action(state)
                
                # Add exploration noise for training
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Reshape for batch processing
                state_batch = np.reshape(state, [1, self.state_dim])
                next_state_batch = np.reshape(next_state, [1, self.state_dim])
                action_batch = np.reshape(action, [1, self.action_dim])
                reward_batch = np.reshape(reward, [1, 1])
                
                # Calculate state values
                v_value = self.critic.model(state_batch)
                next_v_value = self.critic.model(next_state_batch)
                
                # Calculate advantage and TD target
                advantage, td_target = self.calculate_advantage_and_target(
                    reward_batch, v_value, next_v_value, done
                )
                
                # Store in batch
                batch_states.append(state_batch)
                batch_actions.append(action_batch)
                batch_td_targets.append(td_target)
                batch_advantages.append(advantage)
                
                # Update counters
                episode_reward += reward
                episode_length += 1
                
                # Print progress every 100 steps with detailed debug info
                if verbose and episode_length % 100 == 0:
                    print(f"  Step {episode_length}: BGL={info['glucose']:.1f}, "
                          f"Basal={info['basal_insulin']:.2f}, Bolus={info['bolus_insulin']:.2f}, "
                          f"Total={info['total_insulin']:.2f}, "
                          f"PID=[{info['Kp']:.3f}, {info['Ki']:.3f}, {info['Kd']:.3f}], "
                          f"Reward={reward:.2f}")
                
                # Train when batch is full or episode ends
                if len(batch_states) >= self.batch_size or done:
                    if len(batch_states) > 0:
                        # Unpack batches
                        states = self.unpack_batch(batch_states)
                        actions = self.unpack_batch(batch_actions)
                        td_targets = self.unpack_batch(batch_td_targets)
                        advantages = self.unpack_batch(batch_advantages)
                        
                        # Train networks
                        critic_loss = self.critic.train_on_batch(states, td_targets)
                        actor_loss = self.actor.train(states, actions, advantages)
                        
                        # Clear batch
                        batch_states, batch_actions, batch_td_targets, batch_advantages = [], [], [], []
                
                # Update state
                state = next_state
            
            # Episode completed
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            recent_rewards.append(episode_reward)
            
            # Get episode statistics
            stats = self.env.get_statistics()
            self.glucose_stats.append(stats)
            
            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length} minutes")
            print(f"  Mean Glucose: {stats['mean_glucose']:.1f} mg/dL")
            print(f"  Time in Range (80-140): {stats['time_in_range_80_140']:.1f}%")
            print(f"  Time in Range (70-180): {stats['time_in_range_70_180']:.1f}%")
            print(f"  Hypoglycemia (<70): {stats['time_hypo_70']:.1f}%")
            print(f"  Hyperglycemia (>180): {stats['time_hyper_180']:.1f}%")
            print(f"  Total Insulin: {stats['total_insulin']:.1f} U")
            print(f"  Final PID: Kp={stats['final_kp']:.3f}, Ki={stats['final_ki']:.3f}, Kd={stats['final_kd']:.3f}")
            print(f"  Recent 100 avg: {np.mean(recent_rewards):.2f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_weights('best')
                print(f"  *** New best model saved! Reward: {best_reward:.2f} ***")
            
            # Periodic saves
            if (episode + 1) % save_frequency == 0:
                self.save_weights(f'episode_{episode + 1}')
                print(f"  Model saved at episode {episode + 1}")
            
            # Plot progress
            if plot_progress and (episode + 1) % 10 == 0:
                self.plot_training_progress()
            
            # Check for early stopping
            avg_reward = np.mean(recent_rewards)
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            
            if self.episodes_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping after {episode + 1} episodes")
                break
        
        print(f"\nTraining completed! Best reward: {best_reward:.2f}")
        self.save_weights('final')
        
    def test(self, num_episodes=1, render=True, load_best=True):
        """
        Test the trained agent.
        
        Args:
            num_episodes (int): Number of test episodes
            render (bool): Whether to render the environment
            load_best (bool): Whether to load the best saved weights
        """
        if load_best:
            self.load_weights('best')
            print("Loaded best weights for testing")
        
        test_rewards = []
        test_stats = []
        
        for episode in range(num_episodes):
            print(f"\n=== Test Episode {episode + 1}/{num_episodes} ===")
            
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action (no exploration noise during testing)
                action = self.actor.get_action(state)
                
                # Take step
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
            
            test_rewards.append(episode_reward)
            stats = self.env.get_statistics()
            test_stats.append(stats)
            
            print(f"\nTest Episode {episode + 1} Results:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Mean Glucose: {stats['mean_glucose']:.1f} mg/dL")
            print(f"  Time in Range (80-140): {stats['time_in_range_80_140']:.1f}%")
            print(f"  Time in Range (70-180): {stats['time_in_range_70_180']:.1f}%")
            
            # Plot results
            self.env.plot_results()
        
        return test_rewards, test_stats
    
    def save_weights(self, name):
        """Save actor and critic weights."""
        actor_path = os.path.join(self.save_dir, f'diabetes_actor_{name}.h5')
        critic_path = os.path.join(self.save_dir, f'diabetes_critic_{name}.h5')
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
    
    def load_weights(self, name):
        """Load actor and critic weights."""
        actor_path = os.path.join(self.save_dir, f'diabetes_actor_{name}.h5')
        critic_path = os.path.join(self.save_dir, f'diabetes_critic_{name}.h5')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            return True
        return False
    
    def plot_training_progress(self):
        """Plot training progress."""
        if len(self.episode_rewards) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths  
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Time in range
        if self.glucose_stats:
            time_in_range = [stats['time_in_range_80_140'] for stats in self.glucose_stats]
            axes[1, 0].plot(time_in_range)
            axes[1, 0].set_title('Time in Range (80-140 mg/dL)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].grid(True)
            
            # Mean glucose
            mean_glucose = [stats['mean_glucose'] for stats in self.glucose_stats]
            axes[1, 1].plot(mean_glucose)
            axes[1, 1].axhline(y=120, color='g', linestyle='--', label='Target')
            axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].axhline(y=140, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Mean Glucose')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('mg/dL')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        fig.savefig(plot_path)
        plt.close(fig) 