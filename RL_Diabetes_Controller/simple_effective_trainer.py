#!/usr/bin/env python3
"""
Simple and effective diabetes control trainer.
Simplified approach focusing on core learning issues.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimpleDiabetesController:
    """Simple, effective diabetes controller with basic RL."""
    
    def __init__(self):
        """Initialize the simple controller."""
        # Initialize environment
        from envs.diabetes_pid_env import DiabetesPIDEnv
        
        patient_parameters = {
            'BW': 75, 'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
            'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
            'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
            'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
            'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30, 'G_init': 10.0,
            'A_EGP': 0.05, 'phi_EGP': -60, 'F_peak': 1.35,
            'K_rise': 5.0, 'K_decay': 0.01, 'G_thresh': 9.0, 'k_R': 0.0031,
        }
        
        test_case_file = os.path.join('..', 'TestData', 'TestCases.txt')
        self.env = DiabetesPIDEnv(
            patient_params=patient_parameters,
            test_case_file=test_case_file,
            patient_weight=75,
            target_glucose=120
        )
        
        # FIXED: Realistic PID configs for diabetes (much lower Ki values!)
        self.pid_configs = [
            {'Kp': 0.5, 'Ki': 0.001, 'Kd': 0.01},  # Conservative
            {'Kp': 0.8, 'Ki': 0.002, 'Kd': 0.02},  # Moderate  
            {'Kp': 1.0, 'Ki': 0.005, 'Kd': 0.03},  # Aggressive
            {'Kp': 1.2, 'Ki': 0.008, 'Kd': 0.05},  # Very aggressive
            {'Kp': 0.3, 'Ki': 0.001, 'Kd': 0.005}, # Ultra stable
        ]
        
        # Q-values for each glucose range and PID config
        self.glucose_ranges = [
            (0, 70),     # Hypoglycemic
            (70, 80),    # Low normal
            (80, 120),   # Target
            (120, 140),  # High normal  
            (140, 180),  # Mild hyperglycemic
            (180, 300),  # Hyperglycemic
        ]
        
        # Initialize Q-table
        self.q_table = np.zeros((len(self.glucose_ranges), len(self.pid_configs)))
        self.q_counts = np.zeros((len(self.glucose_ranges), len(self.pid_configs)))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.epsilon = 0.3  # Exploration
        self.epsilon_decay = 0.995
        
    def _get_glucose_state(self, glucose):
        """Get discrete glucose state."""
        for i, (low, high) in enumerate(self.glucose_ranges):
            if low <= glucose < high:
                return i
        return len(self.glucose_ranges) - 1  # Default to last range
    
    def _simple_reward(self, glucose):
        """Simple, effective reward function."""
        if 80 <= glucose <= 140:
            return 10  # Good range
        elif 70 <= glucose < 80 or 140 < glucose <= 180:
            return 5   # Acceptable
        elif glucose < 70:
            return -20  # Hypoglycemia penalty
        elif glucose > 180:
            return -10  # Hyperglycemia penalty
        else:
            return -50  # Extreme values
    
    def choose_action(self, glucose_state):
        """Choose PID configuration using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.pid_configs))  # Explore
        else:
            return np.argmax(self.q_table[glucose_state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table with simple Q-learning."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state, action] = new_q
        self.q_counts[state, action] += 1
    
    def run_episode(self, episode_num):
        """Run a single episode with simple control."""
        state = self.env.reset()
        
        # Override environment's PID reset with our simple approach
        self.env.pid.clear()
        self.env.pid.SetPoint = 120
        
        total_reward = 0
        glucose_history = []
        action_history = []
        step = 0
        
        while step < 1440 and not self.env.done:  # 24 hours max
            # Get current glucose
            current_glucose = self.env.patient.G * 18.0182
            glucose_state = self._get_glucose_state(current_glucose)
            
            # Choose PID configuration
            action = self.choose_action(glucose_state)
            pid_config = self.pid_configs[action]
            
            # Apply PID configuration
            self.env.pid.Kp = pid_config['Kp']
            self.env.pid.Ki = pid_config['Ki']
            self.env.pid.Kd = pid_config['Kd']
            
            # FIXED: Force the environment to use our PID values correctly
            dummy_action = np.array([0.0, 0.0, 0.0])  # Dummy action
            
            def fixed_step(action):
                # Skip the RL action and use our configured PID directly
                # Get current glucose
                current_glucose = self.env.patient.G * 18.0182
                
                # Calculate PID output manually
                glucose_error = current_glucose - self.env.target_glucose
                self.env.pid.update(glucose_error)
                
                # Force proper insulin calculation (FIXED)
                pid_output = self.env.pid.output
                basal_insulin = max(0, pid_output * 0.01)  # Proper scaling
                
                # Handle meal bolus
                bolus_insulin = self.env._handle_meal_bolus()
                if bolus_insulin == 0:
                    bolus_insulin = self.env._get_bolus_insulin()
                
                total_insulin = basal_insulin + bolus_insulin
                
                # Apply to patient
                self.env.patient.step(insulin_dose=total_insulin, dt=1.0)
                
                # Update environment state
                self.env.current_step += 1
                self.env.time_since_last_meal += 1
                self.env.time_since_last_insulin += 1
                self.env.previous_glucose = current_glucose
                
                # Store history
                self.env.glucose_history.append(current_glucose)
                self.env.insulin_history.append(total_insulin)
                self.env.pid_history['Kp'].append(self.env.pid.Kp)
                self.env.pid_history['Ki'].append(self.env.pid.Ki)
                self.env.pid_history['Kd'].append(self.env.pid.Kd)
                
                # Check done
                if self.env.current_step >= 1440:
                    self.env.done = True
                
                info = {
                    'glucose': current_glucose,
                    'basal_insulin': basal_insulin,
                    'bolus_insulin': bolus_insulin,  
                    'total_insulin': total_insulin,
                    'Kp': self.env.pid.Kp,
                    'Ki': self.env.pid.Ki,
                    'Kd': self.env.pid.Kd
                }
                
                return self.env._get_state(), 0, self.env.done, info
            
            next_state, env_reward, done, info = fixed_step(dummy_action)
            
            # Use our simple reward
            reward = self._simple_reward(current_glucose)
            total_reward += reward
            
            # Get next glucose state for Q-learning
            next_glucose = info.get('glucose', current_glucose)
            next_glucose_state = self._get_glucose_state(next_glucose)
            
            # Update Q-table
            self.update_q_table(glucose_state, action, reward, next_glucose_state)
            
            # Track history
            glucose_history.append(current_glucose)
            action_history.append(action)
            
            step += 1
            
            # Progress logging
            if step % 200 == 0:
                print(f"    {step:3d} min: BGL={current_glucose:5.1f}, "
                      f"PID={action}({pid_config['Kp']:.1f},{pid_config['Ki']:.2f},{pid_config['Kd']:.2f}), "
                      f"R={reward:3.0f}")
        
        # Calculate episode statistics
        glucose_array = np.array(glucose_history)
        tir_80_140 = np.sum((glucose_array >= 80) & (glucose_array <= 140)) / len(glucose_array) * 100
        tir_70_180 = np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100
        mean_glucose = np.mean(glucose_array)
        
        # Decay exploration
        self.epsilon *= self.epsilon_decay
        
        return {
            'reward': total_reward,
            'tir_80_140': tir_80_140,
            'tir_70_180': tir_70_180,
            'mean_glucose': mean_glucose,
            'steps': step,
            'glucose_history': glucose_history,
            'action_history': action_history
        }
    
    def train(self, num_episodes=100):
        """Train the simple controller."""
        print("🎯 SIMPLE DIABETES CONTROL TRAINING")
        print("="*50)
        print("Approach: Discrete PID selection with Q-learning")
        print(f"Episodes: {num_episodes}")
        print(f"PID Configs: {len(self.pid_configs)}")
        print("="*50)
        
        episode_results = []
        best_tir = 0
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            result = self.run_episode(episode)
            episode_results.append(result)
            
            tir = result['tir_80_140']
            if tir > best_tir:
                best_tir = tir
                print(f"   🎯 NEW BEST TIR: {best_tir:.1f}%")
            
            print(f"✅ Episode {episode + 1}: Reward={result['reward']:6.1f}, "
                  f"TIR={tir:5.1f}%, Mean BGL={result['mean_glucose']:5.1f}, "
                  f"Epsilon={self.epsilon:.3f}")
            
            # Show Q-table learning progress
            if (episode + 1) % 20 == 0:
                self._show_q_table()
                self._plot_progress(episode_results)
        
        print(f"\n🎉 TRAINING COMPLETE! Best TIR: {best_tir:.1f}%")
        return episode_results
    
    def _show_q_table(self):
        """Show current Q-table values."""
        print("\n📊 Q-Table Status:")
        range_names = ["<70", "70-80", "80-120", "120-140", "140-180", ">180"]
        
        for i, range_name in enumerate(range_names):
            q_values = self.q_table[i]
            best_action = np.argmax(q_values)
            best_config = self.pid_configs[best_action]
            print(f"  {range_name:8}: Best=PID{best_action} "
                  f"(Kp={best_config['Kp']:.1f}, Ki={best_config['Ki']:.2f}, Kd={best_config['Kd']:.2f})")
    
    def _plot_progress(self, results):
        """Plot training progress."""
        if len(results) < 5:
            return
        
        episodes = range(1, len(results) + 1)
        tirs = [r['tir_80_140'] for r in results]
        rewards = [r['reward'] for r in results]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, tirs)
        plt.axhline(y=75, color='g', linestyle='--', label='Target 75%')
        plt.title('Time in Range (80-140 mg/dL)')
        plt.xlabel('Episode')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        
        plt.subplot(1, 2, 2)
        plt.plot(episodes, rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def test(self, num_episodes=5):
        """Test the trained controller."""
        print("\n🧪 TESTING SIMPLE CONTROLLER...")
        self.epsilon = 0  # No exploration during testing
        
        test_results = []
        
        for episode in range(num_episodes):
            result = self.run_episode(-1)  # -1 indicates testing
            test_results.append(result)
            
            print(f"  Test {episode + 1}: Reward={result['reward']:6.1f}, "
                  f"TIR(80-140)={result['tir_80_140']:5.1f}%, "
                  f"Mean BGL={result['mean_glucose']:5.1f}")
        
        # Summary
        avg_tir = np.mean([r['tir_80_140'] for r in test_results])
        std_tir = np.std([r['tir_80_140'] for r in test_results])
        avg_reward = np.mean([r['reward'] for r in test_results])
        
        print(f"\n📊 TEST RESULTS:")
        print(f"  Average TIR: {avg_tir:.1f}% ± {std_tir:.1f}%")
        print(f"  Average Reward: {avg_reward:.1f}")
        print(f"  Consistency: {100 - (std_tir/avg_tir*100 if avg_tir > 0 else 0):.1f}%")
        
        return test_results

def main():
    """Main function."""
    controller = SimpleDiabetesController()
    
    try:
        # Train
        training_results = controller.train(num_episodes=80)
        
        # Test
        test_results = controller.test(num_episodes=5)
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 