#!/usr/bin/env python3
"""
Robust training for diabetes control with better handling of early terminations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from A2C.diabetes_a2c_agent import DiabetesA2CAgent
from envs.diabetes_pid_env import DiabetesPIDEnv

class RobustTrainer:
    """Robust trainer that handles early terminations and provides better stability."""
    
    def __init__(self, env):
        """Initialize the robust trainer."""
        self.env = env
        self.agent = DiabetesA2CAgent(env)
        
        # Override some agent settings for more robust training
        self.agent.early_stopping_patience = 200  # Even more patience
        self.agent.exploration_noise = 0.20  # Start with higher exploration
        
        # Training metrics
        self.safety_violations = {'hypo': 0, 'hyper': 0}
        self.episode_types = {'completed': 0, 'terminated': 0}
        self.performance_history = deque(maxlen=50)
        
    def train_robust(self, max_episodes=500):
        """Train with robust handling of safety violations."""
        print("🛡️  ROBUST DIABETES CONTROL TRAINING")
        print("="*60)
        print(f"Max Episodes: {max_episodes}")
        print(f"Early Stopping: {self.agent.early_stopping_patience} episodes")
        print(f"Initial Exploration: {self.agent.exploration_noise}")
        print("="*60)
        
        best_performance = -float('inf')
        consecutive_improvements = 0
        
        for episode in range(max_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Episode tracking
            min_glucose = float('inf')
            max_glucose = -float('inf')
            safety_violations_this_episode = 0
            
            print(f"\n📊 Episode {episode + 1}/{max_episodes}")
            
            # Collect experience batch
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
            
            while not done and episode_length < 1440:  # Max 24 hours
                # Get action with adaptive exploration
                action = self.agent.actor.get_action(state)
                
                # Adaptive noise based on performance
                current_noise = self.agent.exploration_noise
                if len(self.performance_history) > 10:
                    recent_performance = np.mean(list(self.performance_history)[-10:])
                    if recent_performance < 50:  # Poor performance, more exploration
                        current_noise *= 1.5
                    elif recent_performance > 80:  # Good performance, less exploration
                        current_noise *= 0.7
                
                noise = np.random.normal(0, current_noise, size=action.shape)
                action = np.clip(action + noise, -self.agent.action_bound, self.agent.action_bound)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Track glucose levels
                current_glucose = info.get('glucose', 120)
                min_glucose = min(min_glucose, current_glucose)
                max_glucose = max(max_glucose, current_glucose)
                
                # Count safety violations
                if current_glucose < 50 or current_glucose > 250:
                    safety_violations_this_episode += 1
                
                # Store experience
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)
                batch_dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                # Train on mini-batches
                if len(batch_states) >= 16:  # Smaller batches for more frequent updates
                    self._train_batch(batch_states, batch_actions, batch_rewards, 
                                    batch_next_states, batch_dones)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                
                state = next_state
                
                # Progress logging
                if episode_length % 200 == 0:
                    print(f"  ⏱️  {episode_length:4d} min: BGL={current_glucose:5.1f}, "
                          f"Insulin={info.get('total_insulin', 0):4.2f}, "
                          f"PID=[{info.get('Kp', 0):.2f},{info.get('Ki', 0):.2f},{info.get('Kd', 0):.2f}]")
            
            # Final batch training if needed
            if len(batch_states) > 0:
                self._train_batch(batch_states, batch_actions, batch_rewards, 
                                batch_next_states, batch_dones)
            
            # Episode statistics
            stats = self.env.get_statistics()
            tir_80_140 = stats.get('time_in_range_80_140', 0)
            tir_70_180 = stats.get('time_in_range_70_180', 0)
            mean_glucose = stats.get('mean_glucose', 120)
            
            # Track episode type
            if done and episode_length < 1440:
                self.episode_types['terminated'] += 1
                if min_glucose < 50:
                    self.safety_violations['hypo'] += 1
                if max_glucose > 250:
                    self.safety_violations['hyper'] += 1
            else:
                self.episode_types['completed'] += 1
            
            # Store performance
            self.performance_history.append(tir_80_140)
            
            # Episode summary
            print(f"✅ Episode {episode + 1} Complete:")
            print(f"   Reward: {episode_reward:7.1f} | Length: {episode_length:4d} min")
            print(f"   TIR(80-140): {tir_80_140:5.1f}% | TIR(70-180): {tir_70_180:5.1f}%")
            print(f"   Mean BGL: {mean_glucose:5.1f} | Range: {min_glucose:.1f}-{max_glucose:.1f}")
            print(f"   Safety violations: {safety_violations_this_episode}")
            
            # Check for improvement
            if tir_80_140 > best_performance:
                best_performance = tir_80_140
                consecutive_improvements += 1
                self.agent.save_weights(f'robust_best_{episode+1}')
                print(f"   🎯 NEW BEST! TIR: {best_performance:.1f}%")
            else:
                consecutive_improvements = 0
            
            # Adaptive learning
            if episode > 0 and episode % 25 == 0:
                self._adaptive_learning_update(episode)
            
            # Periodic saves
            if (episode + 1) % 50 == 0:
                self.agent.save_weights(f'robust_episode_{episode + 1}')
                self._print_training_summary(episode + 1)
            
            # Progress visualization
            if (episode + 1) % 25 == 0:
                self._plot_robust_progress()
        
        print(f"\n🎉 ROBUST TRAINING COMPLETE!")
        self._print_final_summary(max_episodes)
        return self.agent
    
    def _train_batch(self, states, actions, rewards, next_states, dones):
        """Train on a batch with proper preprocessing."""
        if len(states) == 0:
            return
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Calculate advantages
        advantages = []
        td_targets = []
        
        for i in range(len(states)):
            v_value = self.agent.critic.model(states[i:i+1])[0][0]
            if dones[i]:
                next_v_value = 0
            else:
                next_v_value = self.agent.critic.model(next_states[i:i+1])[0][0]
            
            td_target = rewards[i] + self.agent.gamma * next_v_value
            advantage = td_target - v_value
            
            advantages.append(advantage)
            td_targets.append(td_target)
        
        advantages = np.array(advantages)
        td_targets = np.array(td_targets)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train networks
        try:
            critic_loss = self.agent.critic.train_on_batch(states, td_targets)
            actor_loss = self.agent.actor.train(states, actions, advantages)
        except Exception as e:
            print(f"⚠️  Training error: {e}")
    
    def _adaptive_learning_update(self, episode):
        """Update learning parameters based on performance."""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-10:])
        
        # Adjust exploration based on performance
        if recent_performance < 40:  # Poor performance
            self.agent.exploration_noise = min(0.25, self.agent.exploration_noise * 1.1)
            print(f"   📈 Increased exploration to {self.agent.exploration_noise:.3f}")
        elif recent_performance > 75:  # Good performance
            self.agent.exploration_noise = max(0.05, self.agent.exploration_noise * 0.95)
            print(f"   📉 Reduced exploration to {self.agent.exploration_noise:.3f}")
        
        # Progress report
        print(f"   📊 Recent 10-episode TIR avg: {recent_performance:.1f}%")
    
    def _print_training_summary(self, episode):
        """Print training summary."""
        completed_pct = (self.episode_types['completed'] / episode) * 100
        print(f"\n📈 TRAINING SUMMARY (Episode {episode}):")
        print(f"   Episodes completed: {self.episode_types['completed']}/{episode} ({completed_pct:.1f}%)")
        print(f"   Safety violations: Hypo={self.safety_violations['hypo']}, Hyper={self.safety_violations['hyper']}")
        
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            print(f"   Recent 10-episode TIR avg: {recent_avg:.1f}%")
    
    def _print_final_summary(self, total_episodes):
        """Print final training summary."""
        print(f"📊 FINAL TRAINING SUMMARY:")
        print(f"   Total episodes: {total_episodes}")
        print(f"   Completed episodes: {self.episode_types['completed']}")
        print(f"   Terminated episodes: {self.episode_types['terminated']}")
        print(f"   Success rate: {(self.episode_types['completed']/total_episodes)*100:.1f}%")
        print(f"   Hypoglycemic violations: {self.safety_violations['hypo']}")
        print(f"   Hyperglycemic violations: {self.safety_violations['hyper']}")
        
        if len(self.performance_history) >= 25:
            final_avg = np.mean(list(self.performance_history)[-25:])
            print(f"   Final 25-episode TIR avg: {final_avg:.1f}%")
    
    def _plot_robust_progress(self):
        """Plot training progress."""
        if len(self.performance_history) < 5:
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(list(self.performance_history))
        plt.title('Time in Range (80-140 mg/dL)')
        plt.xlabel('Episode')
        plt.ylabel('Percentage')
        plt.grid(True)
        plt.ylim(0, 100)
        
        plt.subplot(1, 2, 2)
        if len(self.performance_history) >= 10:
            rolling_avg = []
            for i in range(9, len(self.performance_history)):
                rolling_avg.append(np.mean(list(self.performance_history)[i-9:i+1]))
            plt.plot(range(9, len(self.performance_history)), rolling_avg)
            plt.title('10-Episode Rolling Average TIR')
            plt.xlabel('Episode')
            plt.ylabel('Percentage')
            plt.grid(True)
            plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for robust training."""
    
    # Patient parameters
    patient_parameters = {
        'BW': 75,
        'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
        'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
        'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
        'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
        'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
        'G_init': 10.0,
        'A_EGP': 0.05, 'phi_EGP': -60, 'F_peak': 1.35,
        'K_rise': 5.0, 'K_decay': 0.01,
        'G_thresh': 9.0, 'k_R': 0.0031,
    }
    
    # Initialize environment
    test_case_file = os.path.join('..', 'TestData', 'TestCases.txt')
    env = DiabetesPIDEnv(
        patient_params=patient_parameters,
        test_case_file=test_case_file,
        patient_weight=75,
        target_glucose=120
    )
    
    # Create and run robust trainer
    trainer = RobustTrainer(env)
    
    try:
        trained_agent = trainer.train_robust(max_episodes=500)
        
        # Test the final model
        print("\n🧪 TESTING FINAL MODEL...")
        test_rewards, test_stats = trained_agent.test(num_episodes=5, render=True)
        
        print(f"\n🎯 FINAL TEST RESULTS:")
        for i, (reward, stats) in enumerate(zip(test_rewards, test_stats)):
            print(f"   Test {i+1}: Reward={reward:.1f}, "
                  f"TIR(80-140)={stats['time_in_range_80_140']:.1f}%, "
                  f"Mean BGL={stats['mean_glucose']:.1f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        trainer.agent.save_weights('robust_interrupted')
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        trainer.agent.save_weights('robust_error')
        raise

if __name__ == "__main__":
    main() 