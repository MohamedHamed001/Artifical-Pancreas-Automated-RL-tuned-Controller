import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import glob

# Add the parent directories to the path to import required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from utils.pid_controller import PID
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from working_virtual_patient import HovorkaPatient
from utils.insulin_calculator import InsulinCalculator
from utils.meal_parser import MealParser

class DiabetesPIDEnv:
    """
    Diabetes management environment that combines RL-tuned PID control with 
    meal bolus and correction dose calculations.
    """
    
    def __init__(self, patient_params, test_case_file, patient_weight=75, target_glucose=120):
        """
        Initialize the diabetes control environment.
        
        Args:
            patient_params (dict): Hovorka model parameters
            test_case_file (str): Path to test case file with meal/exercise data (used to find TestData directory)
            patient_weight (float): Patient weight in kg
            target_glucose (float): Target glucose level in mg/dL
        """
        # Patient simulation
        self.patient_params = patient_params
        self.patient_weight = patient_weight
        self.target_glucose = target_glucose
        self.patient = None
        
        # Store test case directory info for random selection
        # Find TestData directory (should be in the project root)
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        self.test_data_dir = os.path.join(project_root, 'TestData')
        self.test_cases_file = os.path.join(self.test_data_dir, 'TestCases.txt')
        
        # Find all available test case files
        meal_pattern = os.path.join(self.test_data_dir, 'MealData_case*.data')
        exercise_pattern = os.path.join(self.test_data_dir, 'ExerciseData_case*.data')
        
        self.meal_files = sorted(glob.glob(meal_pattern))
        self.exercise_files = sorted(glob.glob(exercise_pattern))
        
        print(f"Found {len(self.meal_files)} meal test cases and {len(self.exercise_files)} exercise test cases")
        
        # Initialize parser
        self.parser = MealParser()
        
        # Load initial random test case
        self._load_random_test_case()
        
        # Initialize insulin calculator
        self.insulin_calc = InsulinCalculator(patient_weight_kg=patient_weight)
        
        # PID Controller (initialized with reasonable values for diabetes control)
        self.pid = PID(P=0.5, I=0.1, D=0.01)
        self.pid.SetPoint = target_glucose
        self.pid.setSampleTime(1.0)  # 1 minute sampling
        self.pid.setWindup(50.0)  # Prevent integral windup
        
        # Environment parameters
        self.max_episode_length = 1440  # 24 hours in minutes
        self.current_step = 0
        self.done = False
        
        # State space: [BGL, BGL_rate, error, I_term, D_term, Kp, Ki, Kd, 
        #               time_since_meal, time_since_insulin, exercise_status, 
        #               time_of_day_sin, time_of_day_cos]
        self.observation_space = 13
        
        # Action space: [delta_Kp, delta_Ki, delta_Kd] - changes to PID parameters
        self.action_space = 3
        self.action_bound = 0.1  # Maximum change per step (reduced for stability)
        
        # Tracking variables
        self.glucose_history = []
        self.insulin_history = []
        self.pid_history = {'Kp': [], 'Ki': [], 'Kd': []}
        self.reward_history = []
        self.bolus_history = []
        
        # State tracking
        self.previous_glucose = target_glucose
        self.time_since_last_meal = 1440  # Start with no recent meals
        self.time_since_last_insulin = 1440  # Start with no recent insulin
        self.total_episode_reward = 0
        
        # Bolus tracking
        self.bolus_duration = 15  # Bolus delivered over 15 minutes
        self.bolus_remaining = 0  # Units of bolus remaining to deliver
        self.bolus_rate = 0  # Current bolus rate U/h
        
        print(f"Diabetes PID Environment initialized")
        print(f"  Meals: {len(self.meal_data)} events")
        print(f"  Exercise: {len(self.exercise_data)} events") 
        print(f"  Target glucose: {target_glucose} mg/dL")
        print(f"  Patient weight: {patient_weight} kg")
        
    def _load_random_test_case(self):
        """Load a random test case from available files."""
        if self.meal_files and self.exercise_files:
            # Select a random case number
            case_numbers = []
            for meal_file in self.meal_files:
                # Extract case number from filename like "MealData_case42.data"
                basename = os.path.basename(meal_file)
                case_num = basename.replace('MealData_case', '').replace('.data', '')
                case_numbers.append(int(case_num))
            
            # Choose random case
            selected_case = random.choice(case_numbers)
            
            # Load corresponding files
            meal_file = os.path.join(self.test_data_dir, f'MealData_case{selected_case}.data')
            exercise_file = os.path.join(self.test_data_dir, f'ExerciseData_case{selected_case}.data')
            
            # Get body weight for this test case
            self.patient_weight = self._get_body_weight_for_case(selected_case)
            
            # Parse the test case files
            self.meal_data = self._parse_data_file(meal_file, 'meal')
            self.exercise_data = self._parse_data_file(exercise_file, 'exercise')
            
            print(f"Loaded test case {selected_case}: {len(self.meal_data)} meals, {len(self.exercise_data)} exercise events")
            print(f"  Body weight: {self.patient_weight:.1f} kg")
        else:
            print("Warning: No test case files found, using empty data")
            self.meal_data = []
            self.exercise_data = []
    
    def _parse_data_file(self, file_path, data_type):
        """Parse a meal or exercise data file in scientific notation format."""
        data = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 2:
                    time_val = float(parts[0])  # Scientific notation handled automatically
                    if data_type == 'meal':
                        carbs_val = float(parts[1])
                        if carbs_val > 0:  # Only store actual meals, not zero entries
                            data.append({'time': time_val, 'carbs': carbs_val})
                    else:  # exercise
                        active_val = int(float(parts[1]))  # Convert float to int for exercise
                        data.append({'time': time_val, 'active': active_val})
                        
        except FileNotFoundError:
            print(f"Warning: Data file {file_path} not found")
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
            
        return data
    
    def _get_body_weight_for_case(self, case_number):
        """Get body weight for a specific test case from TestCases.txt."""
        try:
            with open(self.test_cases_file, 'r') as f:
                lines = f.readlines()
                
            current_case = None
            for line in lines:
                line = line.strip()
                
                # Look for case header
                if line.startswith(f"Test Case [{case_number}]"):
                    current_case = case_number
                    continue
                
                # If we're in the right case, look for body weight
                if current_case == case_number and line.startswith("Body Weight:"):
                    # Extract weight from line like "Body Weight: 157.28 kg"
                    weight_str = line.split(":")[1].strip().replace(" kg", "")
                    return float(weight_str)
                    
                # If we hit the next case, stop looking
                if current_case == case_number and line.startswith("Test Case ["):
                    break
                    
        except Exception as e:
            print(f"Warning: Could not get body weight for case {case_number}: {e}")
            
        # Default fallback
        return 75.0
        
    def _save_test_case_to_files(self, meal_file_path, exercise_file_path):
        """Save current test case data to temporary files."""
        # Save meal data
        with open(meal_file_path, 'w') as f:
            f.write("# Time(min) Carbs(g)\n")
            for meal in self.meal_data:
                f.write(f"{meal['time']} {meal['carbs']}\n")
                # Add zero entry to indicate meal end
                f.write(f"{meal['time'] + 1} 0.0\n")
        
        print(f"Meal data saved to {meal_file_path}")
        
        # Convert meal data to expected format for logging
        meal_events = [(meal['time'], meal['carbs']) for meal in self.meal_data]
        if meal_events:
            print("Detected meal events (time, carbs):")
            for time, carbs in meal_events:
                print(f"  - {time} min, {carbs} g")
        
        # Save exercise data  
        with open(exercise_file_path, 'w') as f:
            f.write("# Time(min) Active(0/1)\n")
            for exercise in self.exercise_data:
                f.write(f"{exercise['time']} {exercise['active']}\n")
        
        print(f"Exercise data saved to {exercise_file_path}")
        
    def reset(self):
        """Reset the environment for a new episode."""
        # Load a new random test case for this episode
        self._load_random_test_case()
        
        # Initialize patient
        self.patient = HovorkaPatient(patient_params=self.patient_params)
        
        # Create temporary data files from the new test case
        os.makedirs('temp_data', exist_ok=True)
        self._save_test_case_to_files(
            'temp_data/meal_temp.data',
            'temp_data/exercise_temp.data'
        )
        
        # Load data into patient
        self.patient.load_meal_data('temp_data/meal_temp.data')
        self.patient.load_exercise_data('temp_data/exercise_temp.data')
        
        # Reset controllers
        self.pid.clear()
        self.pid.SetPoint = self.target_glucose
        self.pid.Kp = 0.5  # Reset to reasonable values
        self.pid.Ki = 0.1
        self.pid.Kd = 0.01
        self.pid.setWindup(50.0)  # Reset windup guard
        
        # Update insulin calculator with current patient weight
        self.insulin_calc = InsulinCalculator(patient_weight_kg=self.patient_weight)
        
        # Reset environment state
        self.current_step = 0
        self.done = False
        self.previous_glucose = self.target_glucose
        self.time_since_last_meal = 1440
        self.time_since_last_insulin = 1440
        self.total_episode_reward = 0
        
        # Reset bolus tracking
        self.bolus_remaining = 0
        self.bolus_rate = 0
        
        # Clear history
        self.glucose_history = []
        self.insulin_history = []
        self.pid_history = {'Kp': [], 'Ki': [], 'Kd': []}
        self.reward_history = []
        self.bolus_history = []
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation."""
        current_glucose = self.patient.G * 18.0182  # Convert to mg/dL
        glucose_rate = current_glucose - self.previous_glucose  # Rate of change
        error = self.target_glucose - current_glucose
        
        # Time of day (circadian features)
        hour_of_day = (self.patient.time % 1440) / 60.0
        time_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        time_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        
        # Exercise status
        exercise_status = self.patient._get_exercise_status(self.patient.time)
        
        state = np.array([
            current_glucose / 400.0,  # Normalized glucose (0-400 mg/dL range)
            glucose_rate / 100.0,     # Normalized glucose rate
            error / 200.0,            # Normalized error
            self.pid.ITerm / 100.0,   # Normalized integral term
            self.pid.DTerm / 10.0,    # Normalized derivative term
            self.pid.Kp,              # Current Kp
            self.pid.Ki,              # Current Ki  
            self.pid.Kd,              # Current Kd
            min(self.time_since_last_meal / 240.0, 1.0),  # Time since meal (normalized)
            min(self.time_since_last_insulin / 60.0, 1.0), # Time since insulin
            exercise_status,          # Exercise status
            time_sin,                 # Time of day (sin)
            time_cos                  # Time of day (cos)
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, glucose_mgdl, insulin_delivered):
        """Enhanced reward function with stability incentives."""
        reward = 0.0
        
        # Glucose level rewards/penalties with progressive safety
        if glucose_mgdl < 40:  # More severe threshold for termination
            reward = -500  # Severe hypoglycemia - episode termination
            self.done = True
        elif glucose_mgdl > 300:  # More severe threshold for termination
            reward = -500  # Severe hyperglycemia - episode termination  
            self.done = True
        elif glucose_mgdl < 50:
            reward = -200 - (50 - glucose_mgdl) * 10  # Strong penalty but no termination
        elif glucose_mgdl > 250:
            reward = -200 - (glucose_mgdl - 250) * 2  # Strong penalty but no termination
        elif 80 <= glucose_mgdl <= 140:
            # Enhanced rewards for tight control
            if 90 <= glucose_mgdl <= 120:
                reward = 20  # Optimal range bonus
            else:
                reward = 15  # Excellent range
        elif 70 <= glucose_mgdl < 80 or 140 < glucose_mgdl <= 180:
            reward = 5   # Good range
        elif glucose_mgdl < 70:
            reward = -15 - (70 - glucose_mgdl) * 0.8  # Stronger hypoglycemia penalty
        elif glucose_mgdl > 180:
            reward = -10 - (glucose_mgdl - 180) * 0.15  # Stronger hyperglycemia penalty
        
        # Stability bonus - reward consistent glucose levels
        glucose_rate = glucose_mgdl - self.previous_glucose
        if abs(glucose_rate) <= 5:  # Very stable
            reward += 3
        elif abs(glucose_rate) <= 10:  # Moderately stable
            reward += 1
        elif abs(glucose_rate) > 20:  # Rapid change penalty
            reward -= abs(glucose_rate) * 0.2
        
        # Time-in-range streak bonus
        if hasattr(self, 'consecutive_in_range'):
            if 70 <= glucose_mgdl <= 180:
                self.consecutive_in_range += 1
                if self.consecutive_in_range >= 60:  # 1 hour in range
                    reward += 5
            else:
                self.consecutive_in_range = 0
        else:
            self.consecutive_in_range = 1 if 70 <= glucose_mgdl <= 180 else 0
        
        # Penalty for excessive insulin delivery
        if insulin_delivered > 10:  # More than 10 U/h is excessive
            reward -= (insulin_delivered - 10) * 0.8
        
        # PID parameter stability bonus
        kp_change = abs(self.pid.Kp - getattr(self, 'prev_kp', self.pid.Kp))
        ki_change = abs(self.pid.Ki - getattr(self, 'prev_ki', self.pid.Ki))
        kd_change = abs(self.pid.Kd - getattr(self, 'prev_kd', self.pid.Kd))
        
        if kp_change + ki_change + kd_change < 0.1:  # Stable parameters
            reward += 1
        
        # Store previous values for next step
        self.prev_kp = self.pid.Kp
        self.prev_ki = self.pid.Ki
        self.prev_kd = self.pid.Kd
        
        return reward
    
    def _handle_meal_bolus(self):
        """Handle meal bolus delivery if meal is detected."""
        current_meal = self.patient._get_meal_intake(self.patient.time)
        current_glucose = self.patient.G * 18.0182
        
        if current_meal > 0:
            # Deliver meal bolus
            self.insulin_calc.set_current_time(self.patient.time)
            bolus_result = self.insulin_calc.deliver_bolus(
                carbs_grams=current_meal,
                current_glucose_mgdl=current_glucose,
                target_glucose_mgdl=self.target_glucose
            )
            
            if bolus_result['delivered']:
                # Set up bolus delivery over 15 minutes
                self.bolus_remaining = bolus_result['total_dose']
                self.bolus_rate = self.bolus_remaining / self.bolus_duration * 60  # Convert to U/h
                self.time_since_last_meal = 0
                self.time_since_last_insulin = 0
                
                self.bolus_history.append({
                    'time': self.patient.time,
                    'carbs': current_meal,
                    'bolus_dose': bolus_result['bolus_dose'], 
                    'correction_dose': bolus_result['correction_dose'],
                    'total_dose': bolus_result['total_dose']
                })
                
                return self.bolus_rate
        
        return 0.0
    
    def _get_bolus_insulin(self):
        """Get current bolus insulin rate."""
        if self.bolus_remaining > 0:
            # Deliver bolus insulin over time
            bolus_this_minute = min(self.bolus_remaining, self.bolus_rate / 60)
            self.bolus_remaining -= bolus_this_minute
            return bolus_this_minute * 60  # Convert back to U/h
        return 0.0
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (array): [delta_Kp, delta_Ki, delta_Kd] - changes to PID parameters
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Update PID parameters based on RL action with proper bounds
        delta_kp, delta_ki, delta_kd = action
        
        # Apply changes with REALISTIC bounds for diabetes control
        self.pid.Kp = np.clip(self.pid.Kp + delta_kp, 0.01, 2.0)  # Kp: 0.01 to 2.0
        self.pid.Ki = np.clip(self.pid.Ki + delta_ki, 0.0, 0.01)  # Ki: 0.0 to 0.01 (MUCH LOWER!)
        self.pid.Kd = np.clip(self.pid.Kd + delta_kd, 0.0, 0.1)   # Kd: 0.0 to 0.1
        
        # Get current glucose
        current_glucose = self.patient.G * 18.0182
        
        # Handle meal bolus
        meal_bolus_rate = self._handle_meal_bolus()
        
        # Get ongoing bolus insulin
        bolus_rate = self._get_bolus_insulin()
        
        # Calculate PID output for basal insulin (FIXED!)
        if bolus_rate > 0:
            # During bolus, reduce basal but don't turn completely off
            basal_rate = 0.5  # Minimal basal during bolus
        else:
            # FIXED: Use standard PID formula correctly!
            # PID expects SetPoint - ProcessVariable, so:
            self.pid.SetPoint = self.target_glucose  # 120 mg/dL
            self.pid.update(current_glucose)  # Current glucose as feedback
            
            # FIXED: Proper insulin calculation for diabetes
            # Higher glucose error = more insulin needed
            # Use basal rate formula: TDD = weight * 0.55, basal = TDD * 0.5 / 24
            patient_weight = 75  # kg
            estimated_tdd = patient_weight * 0.55  # Total daily dose
            base_basal_rate = estimated_tdd * 0.5 / 24  # 50% of TDD as basal, per hour
            
            # PID adjustment (CORRECTLY handle the sign!)
            # For diabetes: PID output is NEGATIVE when glucose > target
            # We need MORE insulin when glucose is high, so NEGATE the output
            pid_adjustment = -self.pid.output * 0.01  # Negate and scale
            basal_rate = np.clip(base_basal_rate + pid_adjustment, 0.0, 10.0)
        
        # Total insulin delivery
        total_insulin_rate = basal_rate + bolus_rate
        
        # Step the patient simulation
        self.previous_glucose = current_glucose
        new_glucose = self.patient.step(total_insulin_rate)
        
        # Update time tracking
        self.time_since_last_meal += 1
        self.time_since_last_insulin += 1
        if total_insulin_rate > 0:
            self.time_since_last_insulin = 0
        
        # Calculate reward
        reward = self._calculate_reward(new_glucose, total_insulin_rate)
        self.total_episode_reward += reward
        
        # Check episode termination
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            self.done = True
        
        # Record history
        self.glucose_history.append(new_glucose)
        self.insulin_history.append(total_insulin_rate)
        self.pid_history['Kp'].append(self.pid.Kp)
        self.pid_history['Ki'].append(self.pid.Ki)
        self.pid_history['Kd'].append(self.pid.Kd)
        self.reward_history.append(reward)
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'glucose': new_glucose,
            'basal_insulin': basal_rate,
            'bolus_insulin': bolus_rate,
            'total_insulin': total_insulin_rate,
            'Kp': self.pid.Kp,
            'Ki': self.pid.Ki,
            'Kd': self.pid.Kd,
            'episode_reward': self.total_episode_reward,
            'step': self.current_step
        }
        
        return next_state, reward, self.done, info
    
    def render(self, mode='human'):
        """Render the environment state."""
        if not self.glucose_history:
            print("No data to render yet.")
            return
            
        if mode == 'human':
            print(f"Step {self.current_step}: BGL={self.glucose_history[-1]:.1f} mg/dL, "
                  f"Insulin={self.insulin_history[-1]:.2f} U/h, "
                  f"PID=[{self.pid.Kp:.3f}, {self.pid.Ki:.3f}, {self.pid.Kd:.3f}], "
                  f"Reward={self.reward_history[-1]:.2f}")
    
    def plot_results(self):
        """Plot episode results."""
        if not self.glucose_history:
            print("No data to plot.")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        time_points = list(range(len(self.glucose_history)))
        
        # Glucose plot
        axes[0].plot(time_points, self.glucose_history, 'b-', linewidth=2, label='Blood Glucose')
        axes[0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Hypo threshold')
        axes[0].axhline(y=180, color='r', linestyle='--', alpha=0.7, label='Hyper threshold')
        axes[0].axhline(y=self.target_glucose, color='g', linestyle='-', alpha=0.7, label='Target')
        axes[0].fill_between(time_points, 80, 140, alpha=0.2, color='green', label='Excellent range')
        axes[0].set_ylabel('Glucose (mg/dL)')
        axes[0].set_title('Blood Glucose Control')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Insulin plot
        axes[1].plot(time_points, self.insulin_history, 'r-', linewidth=2, label='Total Insulin')
        axes[1].set_ylabel('Insulin (U/h)')
        axes[1].set_title('Insulin Delivery')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # PID parameters plot
        axes[2].plot(time_points, self.pid_history['Kp'], 'g-', label='Kp')
        axes[2].plot(time_points, self.pid_history['Ki'], 'b-', label='Ki')
        axes[2].plot(time_points, self.pid_history['Kd'], 'm-', label='Kd')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_ylabel('PID Parameters')
        axes[2].set_title('PID Parameter Evolution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """Get episode statistics."""
        if not self.glucose_history:
            return {}
            
        glucose_array = np.array(self.glucose_history)
        
        stats = {
            'mean_glucose': np.mean(glucose_array),
            'std_glucose': np.std(glucose_array),
            'time_in_range_80_140': np.sum((glucose_array >= 80) & (glucose_array <= 140)) / len(glucose_array) * 100,
            'time_in_range_70_180': np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array) * 100,
            'time_hypo_70': np.sum(glucose_array < 70) / len(glucose_array) * 100,
            'time_hyper_180': np.sum(glucose_array > 180) / len(glucose_array) * 100,
            'total_episode_reward': self.total_episode_reward,
            'mean_insulin': np.mean(self.insulin_history),
            'total_insulin': np.sum(self.insulin_history) / 60,  # Convert to total units
            'final_kp': self.pid.Kp,
            'final_ki': self.pid.Ki,
            'final_kd': self.pid.Kd,
            'num_boluses': len(self.bolus_history)
        }
        
        return stats

# Clean up temporary files on exit
import atexit
def cleanup_temp_files():
    import shutil
    if os.path.exists('temp_data'):
        shutil.rmtree('temp_data')
        
atexit.register(cleanup_temp_files) 