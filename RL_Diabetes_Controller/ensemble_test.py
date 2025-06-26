#!/usr/bin/env python3
"""
Ensemble testing for diabetes control to improve consistency.
Uses multiple trained models and averages their predictions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from A2C.diabetes_a2c_agent import DiabetesA2CAgent
from envs.diabetes_pid_env import DiabetesPIDEnv

class EnsembleAgent:
    """Ensemble of multiple trained agents for improved consistency."""
    
    def __init__(self, env, model_paths):
        """
        Initialize ensemble with multiple trained models.
        
        Args:
            env: Environment instance
            model_paths: List of paths to trained model weights
        """
        self.env = env
        self.agents = []
        
        # Load multiple agents
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}: {model_path}")
            agent = DiabetesA2CAgent(env)
            try:
                agent.load_weights(model_path)
                self.agents.append(agent)
                print(f"  ✓ Model {i+1} loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load model {i+1}: {e}")
    
    def get_ensemble_action(self, state):
        """Get action by averaging predictions from all models."""
        if not self.agents:
            raise ValueError("No models loaded in ensemble")
        
        actions = []
        for agent in self.agents:
            action = agent.actor.get_action(state)
            actions.append(action)
        
        # Average the actions
        ensemble_action = np.mean(actions, axis=0)
        
        # Add small amount of noise for exploration during testing
        noise = np.random.normal(0, 0.02, size=ensemble_action.shape)
        ensemble_action = np.clip(ensemble_action + noise, -0.1, 0.1)
        
        return ensemble_action
    
    def test_ensemble(self, num_episodes=10, render=True):
        """Test the ensemble across multiple episodes."""
        episode_rewards = []
        episode_stats = []
        
        print(f"\nTesting ensemble with {len(self.agents)} models...")
        print(f"{'='*60}")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1440:  # Max 24 hours
                # Get ensemble action
                action = self.get_ensemble_action(state)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                state = next_state
                step += 1
            
            # Get episode statistics
            stats = self.env.get_statistics()
            episode_rewards.append(episode_reward)
            episode_stats.append(stats)
            
            print(f"Episode {episode + 1:2d}: "
                  f"Reward={episode_reward:7.1f}, "
                  f"TIR(80-140)={stats['time_in_range_80_140']:5.1f}%, "
                  f"Mean BGL={stats['mean_glucose']:5.1f}, "
                  f"Steps={step}")
            
            if render and episode < 3:  # Plot first 3 episodes
                self.env.plot_results()
                plt.title(f'Ensemble Test Episode {episode + 1}')
                plt.show()
        
        # Summary statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_tir = np.mean([s['time_in_range_80_140'] for s in episode_stats])
        std_tir = np.std([s['time_in_range_80_140'] for s in episode_stats])
        
        print(f"\n{'='*60}")
        print("ENSEMBLE TEST SUMMARY:")
        print(f"{'='*60}")
        print(f"Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"Mean TIR (80-140): {mean_tir:.1f}% ± {std_tir:.1f}%")
        print(f"Consistency Score: {100 - (std_tir / mean_tir * 100):.1f}%")
        
        return episode_rewards, episode_stats

def main():
    """Main function for ensemble testing."""
    
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
    
    # Find all saved model weights
    save_dir = os.path.join('A2C', 'save_weights')
    model_paths = []
    
    # Look for best models (you can adjust this based on your saved models)
    potential_models = [
        'diabetes_actor_best',
        'diabetes_actor_episode_500',
        'diabetes_actor_episode_475',
        'diabetes_actor_episode_450',
        'diabetes_actor_final'
    ]
    
    for model_name in potential_models:
        model_path = os.path.join(save_dir, f'{model_name}.h5')
        if os.path.exists(model_path):
            model_paths.append(model_path.replace('.h5', ''))
    
    if not model_paths:
        print("No trained models found! Train the model first.")
        return
    
    print(f"Found {len(model_paths)} trained models for ensemble:")
    for i, path in enumerate(model_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # Create and test ensemble
    ensemble = EnsembleAgent(env, model_paths)
    
    if ensemble.agents:
        ensemble.test_ensemble(num_episodes=10, render=True)
    else:
        print("No models could be loaded for ensemble testing.")

if __name__ == "__main__":
    main() 