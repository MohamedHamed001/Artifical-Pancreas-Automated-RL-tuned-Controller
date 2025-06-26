# -*- coding: utf-8 -*-
"""
Gym-compatible environment wrapper for Hovorka patient simulation
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from collections import deque

from working_virtual_patient import HovorkaPatient

class HovorkaGymEnv(gym.Env):
    """
    Gym-compatible environment for Hovorka patient simulation.
    Action: continuous insulin infusion rate [0, 1] U/h
    Observation: full state (see below)
    """
    def __init__(self, 
                 patient_params: Dict[str, Any],
                 meal_data_path: str,
                 exercise_data_path: str,
                 patient_id: Any = None, # Added for curriculum learning
                 simulation_duration: int = 1440,
                 target_glucose: float = 120.0,
                 reward_scale: float = 1.0):
        super().__init__()
        self.patient_id = patient_id
        self.patient_params = patient_params
        self.meal_data_path = meal_data_path
        self.exercise_data_path = exercise_data_path
        self.simulation_duration = simulation_duration
        self.target_glucose = target_glucose
        self.reward_scale = reward_scale
        
        # Patient initialization is now in reset() to allow for curriculum changes
        self.patient = None
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,))
        
        # State includes history of last 4 observations + 2 new meal info states
        self.history_len = 4
        # The +2 is for time_to_next_meal and carbs_in_next_meal
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15 * self.history_len + 2,))
        self.obs_history = deque(maxlen=self.history_len)

        self.current_step = 0
        self.episode_reward = 0.0
        self.glucose_history = []
        self.insulin_history = []
        self.reward_history = []
        self.target_range = (70.0, 180.0)
        self.last_insulin_rate = 0.0
        self.post_meal_timer = 0 # Timer to track post-meal period
    def _get_observation(self) -> np.ndarray:
        G = self.patient.G
        BGL = G * 18.0182
        current_meal = self.patient._get_meal_intake(self.patient.time)
        current_exercise = self.patient._get_exercise_status(self.patient.time)
        hour_of_day = (self.patient.time % 1440) / 60.0
        time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        observation = np.array([
            BGL,
            self.patient.I,
            self.patient.x1,
            self.patient.x2,
            self.patient.x3,
            self.patient.Q1,
            self.patient.Q2,
            self.patient.D1,
            self.patient.D2,
            current_meal,
            current_exercise,
            self.patient.F_sensitivity,
            time_sin,
            time_cos,
            self.patient.time / 1440.0,
        ], dtype=np.float32)
        return observation
    def _get_meal_info(self) -> Tuple[float, float]:
        """Gets info about the next meal."""
        current_time = self.patient.time
        future_meals = self.patient.meal_data[self.patient.meal_data['time'] > current_time]
        
        if not future_meals.empty:
            next_meal = future_meals.iloc[0]
            time_to_next_meal = next_meal['time'] - current_time
            carbs_in_next_meal = next_meal['carbs']
            return time_to_next_meal, carbs_in_next_meal
        else:
            return 0.0, 0.0 # No more meals
    def _get_stacked_observation(self) -> np.ndarray:
        """Get the full observation including history and meal info."""
        # Update the history buffer with the latest base observation
        self.obs_history.append(self._get_observation())
        
        # Get historical observations
        stacked_obs = np.array(self.obs_history).flatten()
        
        # Get upcoming meal info
        time_to_next_meal, carbs_in_next_meal = self._get_meal_info()
        meal_info = np.array([time_to_next_meal / 60.0, carbs_in_next_meal / 100.0], dtype=np.float32) # Normalize
        
        # Combine historical data and meal info
        full_observation = np.concatenate([stacked_obs, meal_info])
        return full_observation
    def _calculate_reward(self, BGL: float, insulin_rate: float) -> float:
        """
        An improved reward function with better learning signals and intermediate rewards.
        """
        # 1. Safety First: Moderate penalty for hypoglycemia.
        if BGL < 70:
            return -200 # A strong, but not overwhelming, penalty

        # 2. Reward for being in the target range (90-140 mg/dL)
        if 90 <= BGL <= 140:
            return 15.0 # High, flat reward for being in the ideal zone
            
        # 3. Gaussian-like penalty for being outside the target range
        if BGL > 140:
            # Penalty increases the further BGL is from 140
            penalty = 0.1 * (BGL - 140)
        else: # BGL < 90
            # Penalty increases the further BGL is from 90
            penalty = 0.2 * (90 - BGL) # Higher penalty for approaching hypo
            
        # 4. Action Smoothness Penalty (penalize insulin changes)
        insulin_change = abs(insulin_rate - self.last_insulin_rate)
        action_penalty = - (insulin_change ** 2) * 0.1

        # The final reward is a combination of these factors.
        total_reward = -penalty + action_penalty
        
        return total_reward
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # The action is now the direct insulin rate, not clipped here
        insulin_rate = action[0]
        BGL = self.patient.step(insulin_rate)
        
        # The full state is now retrieved by a new helper function
        observation = self._get_stacked_observation()

        # Check for a new meal to start the post-meal timer
        # This now uses the patient's internal meal intake check
        if self.patient._get_meal_intake(self.patient.time) > 0 and self.post_meal_timer == 0:
            self.post_meal_timer = 120 # Start a 2-hour timer
        
        reward = self._calculate_reward(BGL, insulin_rate)

        # Decrement timer if active
        if self.post_meal_timer > 0:
            self.post_meal_timer -= 1
        
        self.current_step += 1
        self.episode_reward += reward
        self.glucose_history.append(BGL)
        self.insulin_history.append(insulin_rate)
        self.reward_history.append(reward)
        self.last_insulin_rate = insulin_rate
        
        # --- Episode Termination Conditions ---
        # End episode if it runs for the full duration or if hypoglycemia occurs
        done = (
            self.current_step >= self.simulation_duration or
            BGL < 50 or
            BGL > 250
        )
        
        info = {
            'BGL': BGL,
            'insulin_rate': insulin_rate,
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'meal_intake': self.patient._get_meal_intake(self.patient.time),
            'exercise_status': self.patient._get_exercise_status(self.patient.time),
            'exercise_sensitivity': self.patient.F_sensitivity
        }
        
        return observation, reward, done, info
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """
        Resets the environment for a new episode.
        Now re-initializes the patient to support curriculum learning.
        """
        # super().reset() # Removed, as gym.Env does not implement this
        
        # Re-initialize the patient
        self.patient = HovorkaPatient(patient_params=self.patient_params)
        self.patient.load_meal_data(self.meal_data_path)
        self.patient.load_exercise_data(self.exercise_data_path)
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.glucose_history = []
        self.insulin_history = []
        self.reward_history = []
        self.last_insulin_rate = 0.0
        self.post_meal_timer = 0
        
        # Reset observation history and populate it with initial state
        self.obs_history.clear()
        initial_obs = self._get_observation()
        for _ in range(self.history_len):
            self.obs_history.append(initial_obs)
            
        # Get the full stacked observation for the initial state
        stacked_obs = self._get_stacked_observation()
        return stacked_obs
    def render(self, mode='human', close=False):
        if close:
            plt.close('all')
            return
        
        if not self.glucose_history:
            print("No simulation data to render. Run simulation first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        time_points = list(range(len(self.glucose_history)))
        ax1.plot(time_points, self.glucose_history, 'b-', linewidth=2, label='Blood Glucose')
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Hypoglycemia threshold')
        ax1.axhline(y=180, color='r', linestyle='--', alpha=0.7, label='Hyperglycemia threshold')
        ax1.axhline(y=self.target_glucose, color='g', linestyle='-', alpha=0.7, label='Target glucose')
        ax1.set_ylabel('Glucose (mg/dL)')
        ax1.set_title('Glucose Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(time_points, self.insulin_history, 'r-', linewidth=2, label='Insulin Rate')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Insulin (U/h)')
        ax2.set_title('Insulin Infusion Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
    def get_statistics(self) -> Dict[str, Any]:
        if not self.glucose_history:
            return {}
        glucose_array = np.array(self.glucose_history)
        stats = {
            'mean_glucose': np.mean(glucose_array),
            'std_glucose': np.std(glucose_array),
            'min_glucose': np.min(glucose_array),
            'max_glucose': np.max(glucose_array),
            'time_in_target': np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100,
            'time_hypo': np.sum(glucose_array < 70) / len(glucose_array) * 100,
            'time_hyper': np.sum(glucose_array > 180) / len(glucose_array) * 100,
            'total_reward': self.episode_reward,
            'mean_insulin': np.mean(self.insulin_history) if self.insulin_history else 0.0
        }
        return stats

def test_gym_environment():
    """
    A simple test function to ensure the gym environment is working correctly.
    """
    print("Testing Hovorka Gym Environment...")
    # Define default patient parameters for testing
    patient_parameters = {
        'BW': 75,
        'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
        'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
        'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
        'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
        'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
        'G_init': 10.0,
        'A_EGP': 0.05,
        'phi_EGP': -60,
        'F_peak': 1.35,
        'K_rise': 5.0,
        'K_decay': 0.01,
        'G_thresh': 9.0,
        'k_R': 0.0031,
    }
    
    # Initialize the environment with specific data files for the test
    env = HovorkaGymEnv(
        patient_params=patient_parameters,
        meal_data_path='Data/MealDataTest.data',
        exercise_data_path='Data/ExerciseData1.data',
        patient_id='test_patient_01',
        simulation_duration=1440 # 24-hour duration for testing
    )
    
    print("--- Testing Gym Environment ---")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Test the reset method
    observation = env.reset()
    print(f"Initial Observation Shape: {observation.shape}")
    assert observation.shape == env.observation_space.shape, "Observation shape mismatch"
    
    # Run a short episode with random actions
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample() # Take a random action
        observation, reward, done, info = env.step(action)
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: BGL={info['BGL']:.2f}, Reward={reward:.2f}")
    
    print(f"Episode finished after {step_count} steps.")
    
    # Get and print final statistics
    stats = env.get_statistics()
    print("\n--- Episode Statistics ---")
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Render the results
    env.render()
    plt.show()

if __name__ == '__main__':
    test_gym_environment()