#!/usr/bin/env python3
"""
Fixed diabetes control trainer with fundamental improvements.
Addresses core learning issues: PID bounds, reward scaling, insulin delivery.
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

class FixedDiabetesTrainer:
    """Fixed trainer addressing fundamental learning issues."""
    
    def __init__(self):
        """Initialize with corrected parameters."""
        # Fixed patient parameters for more stable learning
        self.patient_parameters = {
            'BW': 75,
            'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
            'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
            'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
            'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
            'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
            'G_init': 10.0,  # Start at 180 mg/dL
            'A_EGP': 0.05, 'phi_EGP': -60, 'F_peak': 1.35,
            'K_rise': 5.0, 'K_decay': 0.01,
            'G_thresh': 9.0, 'k_R': 0.0031,
        }
        
        # Initialize environment with fixes
        test_case_file = os.path.join('..', 'TestData', 'TestCases.txt')
        self.env = DiabetesPIDEnv(
            patient_params=self.patient_parameters,
            test_case_file=test_case_file,
            patient_weight=75,
            target_glucose=120
        )
        
        # Apply critical fixes to environment
        self._fix_environment()
        
        # Create agent with better parameters
        self.agent = DiabetesA2CAgent(self.env)
        self._fix_agent()
        
    def _fix_environment(self):
        """Apply critical fixes to the environment."""
        print("🔧 Applying environment fixes...")
        
        # Fix 1: Better PID parameter bounds and initialization
        def fixed_reset(self):
            # Original reset logic
            self._load_random_test_case()
            self.patient = __import__('hovorka_gym_env').HovorkaPatient(patient_params=self.patient_params)
            
            os.makedirs('temp_data', exist_ok=True)
            self._save_test_case_to_files('temp_data/meal_temp.data', 'temp_data/exercise_temp.data')
            self.patient.load_meal_data('temp_data/meal_temp.data')
            self.patient.load_exercise_data('temp_data/exercise_temp.data')
            
            # FIXED: Better PID initialization and bounds
            self.pid.clear()
            self.pid.SetPoint = self.target_glucose
            self.pid.Kp = 0.8   # Better starting values
            self.pid.Ki = 0.05  # Lower Ki to prevent windup
            self.pid.Kd = 0.1   # Reasonable Kd
            self.pid.setWindup(20.0)  # Tighter windup guard
            
            # Reset other state
            self.insulin_calc = __import__('utils.insulin_calculator', fromlist=['InsulinCalculator']).InsulinCalculator(patient_weight_kg=self.patient_weight)
            self.current_step = 0
            self.done = False
            self.previous_glucose = self.target_glucose
            self.time_since_last_meal = 1440
            self.time_since_last_insulin = 1440
            self.total_episode_reward = 0
            self.bolus_remaining = 0
            self.bolus_rate = 0
            
            # Clear history
            self.glucose_history = []
            self.insulin_history = []
            self.pid_history = {'Kp': [], 'Ki': [], 'Kd': []}
            self.reward_history = []
            self.bolus_history = []
            
            return self._get_state()
        
        # Fix 2: Better reward function
        def fixed_calculate_reward(self, glucose_mgdl, insulin_delivered):
            """Fixed reward function with reasonable scaling."""
            reward = 0.0
            
            # Base glucose rewards (reasonable scale)
            if glucose_mgdl < 40:
                reward = -100  # Severe but not extreme
                self.done = True
            elif glucose_mgdl > 300:
                reward = -100  # Severe but not extreme
                self.done = True
            elif 80 <= glucose_mgdl <= 140:
                reward = 10  # Good range
                if 90 <= glucose_mgdl <= 120:
                    reward = 15  # Optimal range
            elif 70 <= glucose_mgdl < 80:
                reward = 5   # Acceptable low
            elif 140 < glucose_mgdl <= 180:
                reward = 5   # Acceptable high
            elif glucose_mgdl < 70:
                reward = -5 - (70 - glucose_mgdl) * 0.5  # Gentle penalty
            elif glucose_mgdl > 180:
                reward = -5 - (glucose_mgdl - 180) * 0.1  # Gentle penalty
            
            # Stability bonus
            glucose_rate = abs(glucose_mgdl - self.previous_glucose)
            if glucose_rate <= 3:
                reward += 2  # Stability bonus
            elif glucose_rate > 15:
                reward -= glucose_rate * 0.1  # Instability penalty
            
            # Reasonable insulin penalty
            if insulin_delivered > 15:
                reward -= (insulin_delivered - 15) * 0.2
            
            return reward
        
        # Fix 3: Better PID step function
        def fixed_step(self, action):
            """Fixed step function with proper PID handling."""
            # FIXED: Better action scaling and bounds
            delta_kp, delta_ki, delta_kd = action
            
            # Smaller, more reasonable changes
            self.pid.Kp = np.clip(self.pid.Kp + delta_kp * 0.1, 0.1, 3.0)    # Kp: 0.1 to 3.0
            self.pid.Ki = np.clip(self.pid.Ki + delta_ki * 0.01, 0.0, 0.5)   # Ki: 0.0 to 0.5 (much lower!)
            self.pid.Kd = np.clip(self.pid.Kd + delta_kd * 0.05, 0.0, 0.5)   # Kd: 0.0 to 0.5
            
            # Get current glucose
            current_glucose = self.patient.G * 18.0182
            
            # Handle meal bolus first
            bolus_insulin = self._handle_meal_bolus()
            if bolus_insulin == 0:
                bolus_insulin = self._get_bolus_insulin()
            
            # PID for basal insulin - FIXED calculation
            glucose_error = current_glucose - self.target_glucose
            self.pid.update(glucose_error)
            
            # FIXED: Proper PID output handling
            pid_output = self.pid.output
            basal_insulin = max(0, -pid_output * 0.01)  # Convert to reasonable insulin rate
            
            # Total insulin
            total_insulin = basal_insulin + bolus_insulin
            
            # Apply insulin to patient
            self.patient.step(insulin_dose=total_insulin, dt=1.0)
            
            # Calculate reward
            reward = self.fixed_calculate_reward(current_glucose, total_insulin)
            
            # Update tracking
            self.current_step += 1
            self.time_since_last_meal += 1
            self.time_since_last_insulin += 1
            self.previous_glucose = current_glucose
            
            # Store history
            self.glucose_history.append(current_glucose)
            self.insulin_history.append(total_insulin)
            self.pid_history['Kp'].append(self.pid.Kp)
            self.pid_history['Ki'].append(self.pid.Ki)
            self.pid_history['Kd'].append(self.pid.Kd)
            self.reward_history.append(reward)
            
            # Check termination
            if self.current_step >= 1440 or self.done:
                self.done = True
            
            # Info
            info = {
                'glucose': current_glucose,
                'basal_insulin': basal_insulin,
                'bolus_insulin': bolus_insulin,
                'total_insulin': total_insulin,
                'Kp': self.pid.Kp,
                'Ki': self.pid.Ki,
                'Kd': self.pid.Kd,
                'step': self.current_step
            }
            
            return self._get_state(), reward, self.done, info
        
        # Apply fixes by monkey-patching
        import types
        self.env.reset = types.MethodType(fixed_reset, self.env)
        self.env.step = types.MethodType(fixed_step, self.env)
        self.env.fixed_calculate_reward = types.MethodType(fixed_calculate_reward, self.env)
        
        print("✅ Environment fixes applied")
    
    def _fix_agent(self):
        """Apply fixes to the agent."""
        print("🔧 Applying agent fixes...")
        
        # Better hyperparameters
        self.agent.action_bound = 0.5  # Larger action space
        self.agent.exploration_noise = 0.1  # Lower noise for stability
        self.agent.actor_lr = 0.0001  # Much lower learning rate
        self.agent.critic_lr = 0.0005  # Much lower learning rate
        self.agent.early_stopping_patience = 300  # More patience
        
        print("✅ Agent fixes applied")
    
    def train_fixed(self, max_episodes=300):
        """Train with all fixes applied."""
        print("\n🩺 FIXED DIABETES CONTROL TRAINING")
        print("="*60)
        print("🔧 Fixes Applied:")
        print("  ✅ PID bounds: Kp(0.1-3.0), Ki(0.0-0.5), Kd(0.0-0.5)")
        print("  ✅ Reasonable reward scaling (-100 to +15)")
        print("  ✅ Better insulin delivery calculation")
        print("  ✅ Lower learning rates for stability")
        print("  ✅ Improved action scaling")
        print("="*60)
        
        episode_rewards = []
        episode_tirs = []
        best_tir = 0
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"\n📊 Episode {episode + 1}/{max_episodes}")
            
            while not done and step < 1440:
                # Get action
                action = self.agent.actor.get_action(state)
                
                # Add small exploration noise
                noise = np.random.normal(0, self.agent.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -self.agent.action_bound, self.agent.action_bound)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                # Simple training (no batching for stability)
                if step > 0:  # Skip first step
                    self._simple_train_step(state, action, reward, next_state, done)
                
                state = next_state
                step += 1
                
                # Progress logging
                if step % 240 == 0:  # Every 4 hours
                    print(f"  ⏱️  {step:3d} min: BGL={info['glucose']:5.1f}, "
                          f"Insulin={info['total_insulin']:4.2f}, "
                          f"PID=[{info['Kp']:.2f},{info['Ki']:.3f},{info['Kd']:.2f}], "
                          f"R={reward:5.1f}")
            
            # Episode statistics
            stats = self.env.get_statistics()
            tir = stats.get('time_in_range_80_140', 0)
            mean_glucose = stats.get('mean_glucose', 120)
            
            episode_rewards.append(episode_reward)
            episode_tirs.append(tir)
            
            print(f"✅ Episode {episode + 1}: Reward={episode_reward:6.1f}, "
                  f"TIR={tir:5.1f}%, Mean BGL={mean_glucose:5.1f}")
            
            # Save best model
            if tir > best_tir:
                best_tir = tir
                self.agent.save_weights(f'fixed_best')
                print(f"   🎯 NEW BEST TIR: {best_tir:.1f}%")
            
            # Progress plots
            if (episode + 1) % 25 == 0:
                self._plot_fixed_progress(episode_rewards, episode_tirs)
            
            # Save checkpoints
            if (episode + 1) % 50 == 0:
                self.agent.save_weights(f'fixed_ep_{episode + 1}')
        
        print(f"\n🎉 TRAINING COMPLETE! Best TIR: {best_tir:.1f}%")
        return self.agent
    
    def _simple_train_step(self, state, action, reward, next_state, done):
        """Simple single-step training for stability."""
        try:
            # Reshape inputs
            state = np.reshape(state, [1, -1])
            next_state = np.reshape(next_state, [1, -1])
            action = np.reshape(action, [1, -1])
            
            # Calculate values
            v_current = self.agent.critic.model(state)[0][0]
            v_next = 0 if done else self.agent.critic.model(next_state)[0][0]
            
            # TD target and advantage
            td_target = reward + self.agent.gamma * v_next
            advantage = td_target - v_current
            
            # Train networks (with error handling)
            td_target_array = np.array([[td_target]])
            advantage_array = np.array([[advantage]])
            
            self.agent.critic.train_on_batch(state, td_target_array)
            self.agent.actor.train(state, action, advantage_array)
            
        except Exception as e:
            print(f"⚠️  Training step error: {e}")
    
    def _plot_fixed_progress(self, rewards, tirs):
        """Plot training progress."""
        if len(rewards) < 5:
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards (Fixed)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(tirs)
        plt.axhline(y=75, color='g', linestyle='--', label='Target 75%')
        plt.title('Time in Range (80-140 mg/dL)')
        plt.xlabel('Episode')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
    
    def test_fixed(self, num_episodes=5):
        """Test the fixed model."""
        print("\n🧪 TESTING FIXED MODEL...")
        
        # Load best model
        try:
            self.agent.load_weights('fixed_best')
            print("✅ Loaded best fixed model")
        except:
            print("⚠️  Using current weights")
        
        test_results = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1440:
                # Pure exploitation - no noise
                action = self.agent.actor.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                step += 1
            
            stats = self.env.get_statistics()
            test_results.append({
                'reward': episode_reward,
                'tir': stats.get('time_in_range_80_140', 0),
                'mean_glucose': stats.get('mean_glucose', 120)
            })
            
            print(f"  Test {episode + 1}: Reward={episode_reward:6.1f}, "
                  f"TIR={stats['time_in_range_80_140']:5.1f}%, "
                  f"Mean BGL={stats['mean_glucose']:5.1f}")
        
        # Summary
        avg_tir = np.mean([r['tir'] for r in test_results])
        std_tir = np.std([r['tir'] for r in test_results])
        
        print(f"\n📊 FIXED MODEL RESULTS:")
        print(f"  Average TIR: {avg_tir:.1f}% ± {std_tir:.1f}%")
        print(f"  Consistency: {100 - (std_tir/avg_tir*100):.1f}%")
        
        return test_results

def main():
    """Main function."""
    trainer = FixedDiabetesTrainer()
    
    try:
        # Train the fixed model
        trained_agent = trainer.train_fixed(max_episodes=200)
        
        # Test the results
        test_results = trainer.test_fixed(num_episodes=5)
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted")
        trainer.agent.save_weights('fixed_interrupted')
    except Exception as e:
        print(f"\n❌ Error: {e}")
        trainer.agent.save_weights('fixed_error')
        raise

if __name__ == "__main__":
    main() 