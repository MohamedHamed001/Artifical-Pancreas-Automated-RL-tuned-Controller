#!/usr/bin/env python3
"""
Advanced training for diabetes control with curriculum learning and improved stability.
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

class AdvancedTrainer:
    """Advanced trainer with curriculum learning and stability improvements."""
    
    def __init__(self, env):
        """Initialize the advanced trainer."""
        self.env = env
        self.agent = DiabetesA2CAgent(env)
        
        # Curriculum learning stages
        self.curriculum_stages = [
            {"name": "Easy", "episodes": 100, "complexity": 0.3},
            {"name": "Medium", "episodes": 200, "complexity": 0.6},
            {"name": "Hard", "episodes": 200, "complexity": 1.0}
        ]
        
        # Training tracking
        self.stage_results = []
        self.best_models = []
        
    def adjust_environment_complexity(self, complexity):
        """Adjust environment difficulty based on curriculum stage."""
        # Modify patient parameters for curriculum learning
        if complexity < 0.5:  # Easy stage
            # More forgiving parameters
            self.env.patient_params['k_e'] = 0.18  # Faster insulin clearance
            self.env.patient_params['EGP_0'] = 0.012  # Lower glucose production
        elif complexity < 0.8:  # Medium stage  
            # Normal parameters
            self.env.patient_params['k_e'] = 0.138
            self.env.patient_params['EGP_0'] = 0.0161
        else:  # Hard stage
            # More challenging parameters
            self.env.patient_params['k_e'] = 0.10  # Slower insulin clearance
            self.env.patient_params['EGP_0'] = 0.020  # Higher glucose production
            
    def train_with_curriculum(self):
        """Train using curriculum learning approach."""
        print("🎓 ADVANCED CURRICULUM TRAINING")
        print("="*60)
        
        total_episodes = 0
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            print(f"\n📚 Stage {stage_idx + 1}: {stage['name']} Training")
            print(f"Complexity: {stage['complexity']:.1f}, Episodes: {stage['episodes']}")
            print("-" * 40)
            
            # Adjust environment for this stage
            self.adjust_environment_complexity(stage['complexity'])
            
            # Adjust learning parameters based on stage
            if stage_idx == 0:  # Easy stage - higher exploration
                self.agent.exploration_noise = 0.20
                self.agent.actor_lr = 0.001
            elif stage_idx == 1:  # Medium stage - balanced
                self.agent.exploration_noise = 0.15
                self.agent.actor_lr = 0.0005
            else:  # Hard stage - lower exploration, fine-tuning
                self.agent.exploration_noise = 0.10
                self.agent.actor_lr = 0.0002
            
            # Train for this stage
            stage_rewards = []
            for episode in range(stage['episodes']):
                # Run episode
                state = self.env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                # Collect experience for batch training
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                
                while not done and step < 1440:  # Max 24 hours
                    # Get action with stage-appropriate exploration
                    action = self.agent.actor.get_action(state)
                    noise = np.random.normal(0, self.agent.exploration_noise, size=action.shape)
                    action = np.clip(action + noise, -self.agent.action_bound, self.agent.action_bound)
                    
                    # Take step
                    next_state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    
                    # Store experience
                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_rewards.append(reward)
                    batch_next_states.append(next_state)
                    batch_dones.append(done)
                    
                    state = next_state
                    step += 1
                    
                    # Train on batch
                    if len(batch_states) >= self.agent.batch_size or done:
                        self._train_batch(batch_states, batch_actions, batch_rewards, 
                                        batch_next_states, batch_dones)
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                
                stage_rewards.append(episode_reward)
                total_episodes += 1
                
                # Progress reporting
                if (episode + 1) % 25 == 0:
                    recent_avg = np.mean(stage_rewards[-25:])
                    stats = self.env.get_statistics()
                    print(f"  Episode {episode + 1:3d}/{stage['episodes']:3d}: "
                          f"Avg Reward={recent_avg:6.1f}, "
                          f"TIR={stats['time_in_range_80_140']:5.1f}%")
                
                # Save best model for this stage
                if episode_reward > max(stage_rewards[:-1] + [-float('inf')]):
                    self.agent.save_weights(f'stage_{stage_idx + 1}_best')
            
            # Stage summary
            avg_reward = np.mean(stage_rewards[-50:])  # Last 50 episodes
            self.stage_results.append({
                'stage': stage['name'],
                'avg_reward': avg_reward,
                'episodes': stage['episodes']
            })
            
            print(f"✅ {stage['name']} stage complete! Avg reward: {avg_reward:.1f}")
            
            # Decay exploration noise
            self.agent.exploration_noise *= self.agent.noise_decay
        
        print(f"\n🎉 CURRICULUM TRAINING COMPLETE!")
        print(f"Total episodes: {total_episodes}")
        
        return self.stage_results
    
    def _train_batch(self, states, actions, rewards, next_states, dones):
        """Train on a batch of experiences."""
        if len(states) == 0:
            return
            
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Calculate advantages and targets
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
        actor_loss = self.agent.actor.train(states, actions, advantages)
        critic_loss = self.agent.critic.train(states, td_targets)
    
    def final_evaluation(self, num_episodes=10):
        """Final evaluation with comprehensive testing."""
        print("\n🔬 FINAL EVALUATION")
        print("="*50)
        
        # Load best overall model
        try:
            self.agent.load_weights('stage_3_best')  # Hard stage best
            print("Loaded best model from hard stage")
        except:
            print("Using current model weights")
        
        episode_rewards = []
        episode_stats = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1440:
                # Pure exploitation - no exploration noise
                action = self.agent.actor.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                step += 1
            
            stats = self.env.get_statistics()
            episode_rewards.append(episode_reward)
            episode_stats.append(stats)
            
            print(f"Test {episode + 1:2d}: "
                  f"Reward={episode_reward:7.1f}, "
                  f"TIR(80-140)={stats['time_in_range_80_140']:5.1f}%, "
                  f"Mean BGL={stats['mean_glucose']:5.1f}")
        
        # Summary
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_tir = np.mean([s['time_in_range_80_140'] for s in episode_stats])
        std_tir = np.std([s['time_in_range_80_140'] for s in episode_stats])
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"Mean TIR (80-140): {mean_tir:.1f}% ± {std_tir:.1f}%")
        print(f"Consistency: {100 - (std_tir / mean_tir * 100):.1f}%")
        
        if mean_tir > 75 and std_tir < 10:
            print("🎯 EXCELLENT PERFORMANCE ACHIEVED!")
        elif mean_tir > 65:
            print("✅ Good performance - consider more training")
        else:
            print("⚠️  Performance needs improvement")
        
        return episode_rewards, episode_stats

def main():
    """Main function for advanced training."""
    
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
    
    # Create and run advanced trainer
    trainer = AdvancedTrainer(env)
    
    try:
        # Run curriculum training
        stage_results = trainer.train_with_curriculum()
        
        # Final evaluation
        trainer.final_evaluation(num_episodes=10)
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        trainer.agent.save_weights('advanced_interrupted')
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        trainer.agent.save_weights('advanced_error')
        raise

if __name__ == "__main__":
    main() 