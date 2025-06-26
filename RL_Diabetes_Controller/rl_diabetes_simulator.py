#!/usr/bin/env python3
"""
RL-Tuned PID Diabetes Simulator for External Artificial Pancreas System
Adapter for integrating the trained RL diabetes control model with external systems.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
from dataclasses import dataclass

# Add paths for our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import the PID controller from your utils
from utils.pid_controller import PID

class RLDiabetesSimulator:
    """
    RL-Tuned PID Diabetes Simulator for external artificial pancreas systems.
    Integrates the trained A2C agent with external system interfaces.
    """
    
    def __init__(self, actor_model_path, test_case_path, totalSimulationTimeInNs):
        """
        Initialize the RL diabetes simulator.
        
        Args:
            actor_model_path (str): Path to the trained actor model (diabetes_actor_best.h5)
            test_case_path (str): Path to the test case data file
            totalSimulationTimeInNs (int): Total simulation time in nanoseconds
        """
        print("🩺 Initializing RL-Tuned PID Diabetes Simulator...")
        
        # Load the trained RL actor model
        self.actor_model = self.load_actor_model(actor_model_path)
        
        # Parse test case file
        self.weight, self.meals = self.parse_test_case(test_case_path)
        
        # Convert simulation time
        self.sim_time = int(totalSimulationTimeInNs / 10**9)
        
        # Create meal schedule
        self.meal_schedule = {time: carbs for time, carbs in self.meals}
        
        # Initialize PID controller with realistic diabetes bounds
        self.pid = PID(P=1.0, I=0.002, D=0.05)  # Initial values - will be tuned by RL
        self.pid.SetPoint = 120  # Target glucose
        self.pid.setWindup(20.0)  # Prevent integral windup
        
        # Initialize state
        self.state = self.initialize_state()
        
        print(f"✅ RL Diabetes Simulator initialized")
        print(f"   Patient weight: {self.weight} kg")
        print(f"   Simulation time: {self.sim_time} minutes")
        print(f"   Meal events: {len(self.meals)}")
        
    def load_actor_model(self, path):
        """Load the trained A2C actor model by reconstructing architecture and loading weights."""
        try:
            print(f"🔧 Reconstructing actor model architecture...")
            
            # Recreate actor architecture (matching diabetes_a2c_actor.py)
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Lambda
            
            # Model parameters (should match training)
            state_dim = 13  # Based on diabetes environment
            action_dim = 3  # [delta_Kp, delta_Ki, delta_Kd]
            action_bound = 0.1
            
            # Build actor network (same as DiabetesActor._build_network)
            state_input = Input(shape=(state_dim,))
            
            # Enhanced hidden layers with dropout
            h1 = Dense(128, activation='relu')(state_input)
            h1_dropout = tf.keras.layers.Dropout(0.2)(h1)
            
            h2 = Dense(128, activation='relu')(h1_dropout)
            h2_dropout = tf.keras.layers.Dropout(0.2)(h2)
            
            h3 = Dense(64, activation='relu')(h2_dropout)
            h3_dropout = tf.keras.layers.Dropout(0.1)(h3)
            
            h4 = Dense(32, activation='relu')(h3_dropout)
            
            # Output layer for PID parameter changes
            delta_pid = Dense(action_dim, activation='tanh')(h4)
            scaled_output = Lambda(lambda x: x * action_bound)(delta_pid)
            
            model = Model(inputs=state_input, outputs=scaled_output)
            
            # Load weights
            model.load_weights(path)
            
            print(f"✅ Successfully loaded RL actor weights from: {path}")
            print(f"   State dim: {state_dim}, Action dim: {action_dim}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading actor model: {e}")
            print(f"⚠️  Attempting fallback loading...")
            
            # Fallback: try to create a simple model for testing
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense
                
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(13,)),
                    Dense(32, activation='relu'),
                    Dense(3, activation='tanh')
                ])
                
                print(f"⚠️  Using fallback model architecture")
                return model
                
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                raise Exception(f"Could not load actor model: {e}")
    
    def parse_test_case(self, file_path):
        """Parse test case file to extract weight and meal schedule."""
        print(f"📋 Parsing test case: {file_path}")
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except:
            # Fallback to default test case path
            fallback_path = os.path.join('..', 'TestData', 'TestCases.txt')
            print(f"⚠️  Fallback to: {fallback_path}")
            with open(fallback_path, 'r') as file:
                lines = file.readlines()
        
        weight = None
        meals = []
        
        for line in lines:
            if "Body Weight" in line:
                weight = float(line.strip().split(':')[1].split()[0])
            elif "Meal" in line and "Time" in line:
                parts = line.strip().split(',')
                time = int(parts[0].split(':')[1].strip().split()[0])
                carbs = float(parts[1].split(':')[1].strip().split()[0])
                meals.append((time, carbs))
        
        # Use default values if not found
        if weight is None:
            weight = 75.0
            print("⚠️  Using default weight: 75 kg")
        
        return weight, meals
    
    def initialize_state(self):
        """Initialize simulation state variables."""
        TDI = 0.45 * self.weight  # Total Daily Insulin
        daily_basal_limit = 0.5 * TDI
        
        return {
            "TDI": TDI,
            "daily_basal_limit": daily_basal_limit,
            "previous_glucose": 120.0,
            "cumulative_error": 0.0,
            "next_basal_timestep": 1,
            "basal_tracker": {},
            "basal_rates": [],
            "bolus_insulin": [],
            "carb_intake": [],
            "infusion_values": [],
            "bolusHoldUntil": -1,
            # RL-specific state
            "glucose_history": [120.0] * 5,  # Keep recent history
            "insulin_history": [1.0] * 5,
            "time_since_last_meal": 1440,
            "time_since_last_insulin": 1440,
        }
    
    def get_rl_state(self, glucose, timestep):
        """Get state vector for RL model input (matches training format)."""
        # Update glucose history
        self.state["glucose_history"].append(glucose)
        if len(self.state["glucose_history"]) > 5:
            self.state["glucose_history"].pop(0)
        
        # Time features (cyclic encoding)
        hour = timestep % 24
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)
        
        # Check for upcoming meals
        upcoming_meal = 0
        for meal_time, carbs in self.meals:
            if 0 <= meal_time - timestep <= 60:  # Meal within next hour
                upcoming_meal = carbs
                break
        
        # Construct state vector (13 features to match training)
        state = np.array([
            glucose / 180.0,  # Normalized current glucose
            self.state["previous_glucose"] / 180.0,  # Normalized previous glucose
            (glucose - 120) / 60.0,  # Normalized glucose error
            self.pid.Kp / 3.0,  # Normalized current PID Kp
            self.pid.Ki / 0.01,  # Normalized current PID Ki  
            self.pid.Kd / 0.5,  # Normalized current PID Kd
            time_sin,  # Cyclic time encoding
            time_cos,
            self.weight / 100.0,  # Normalized weight
            upcoming_meal / 100.0,  # Normalized upcoming meal
            self.state["time_since_last_meal"] / 1440.0,  # Normalized time since meal
            self.state["time_since_last_insulin"] / 120.0,  # Normalized time since insulin
            len(self.state["infusion_values"]) / 1440.0  # Normalized simulation progress
        ])
        
        return state.reshape(1, -1)
    
    def predict_pid_parameters(self, glucose, timestep):
        """Use RL model to predict optimal PID parameters."""
        try:
            # Get current state
            rl_state = self.get_rl_state(glucose, timestep)
            
            # Predict action using trained actor
            action = self.actor_model(rl_state, training=False)[0].numpy()
            
            # Convert action to PID parameter changes (small adjustments)
            action_bound = 0.1  # Same as training
            delta_kp, delta_ki, delta_kd = action * action_bound
            
            # Apply PID parameter updates with realistic bounds
            self.pid.setKp(np.clip(self.pid.Kp + delta_kp, 0.1, 3.0))
            self.pid.setKi(np.clip(self.pid.Ki + delta_ki, 0.0, 0.01))  # CRITICAL: Low Ki bound
            self.pid.setKd(np.clip(self.pid.Kd + delta_kd, 0.0, 0.5))
            
            return self.pid.Kp, self.pid.Ki, self.pid.Kd
            
        except Exception as e:
            print(f"⚠️  RL prediction error: {e}")
            # Fallback to safe PID values
            return 1.0, 0.002, 0.05
    
    def predict_insulin_dosage(self, glucose, timestep):
        """Predict insulin dosage using RL-tuned PID controller."""
        # Get RL-optimized PID parameters
        Kp, Ki, Kd = self.predict_pid_parameters(glucose, timestep)
        
        # Apply PID control for basal insulin
        basal_rate = self.adjust_basal_insulin(glucose, Kp, Ki, Kd, timestep)
        
        return basal_rate
    
    def adjust_basal_insulin(self, glucose, Kp, Ki, Kd, timestep):
        """Calculate basal insulin using PID controller."""
        target_glucose = 120
        
        # Update PID controller with current glucose (it calculates error internally)
        self.pid.update(glucose)
        pid_output = self.pid.output
        
        # Base basal rate (physiologically based)
        base_basal_rate = (self.weight * 0.55) * 0.5 / 24  # ~0.86 U/h for 75kg patient
        
        # Apply PID adjustment (CRITICAL: Negate for diabetes control)
        adjusted_rate = base_basal_rate + (-pid_output * 0.01)  # Proper sign handling
        
        # Apply safety limits
        adjusted_rate = np.clip(adjusted_rate, 0.1, 3.0)
        
        # Update state
        self.state["previous_glucose"] = glucose
        
        # Debug output
        error = glucose - target_glucose
        print(f"\n🔧 RL-Tuned PID Control:")
        print(f"   Timestep: {timestep}")
        print(f"   Glucose: {glucose:.1f} mg/dL")
        print(f"   PID: Kp={Kp:.3f}, Ki={Ki:.4f}, Kd={Kd:.3f}")
        print(f"   Error: {error:.1f}, PID Output: {pid_output:.3f}")
        print(f"   Basal Rate: {adjusted_rate:.2f} U/h")
        print("-" * 50)
        
        return adjusted_rate
    
    def calculate_bolus(self, glucose, meal_carbs, timestep, target_glucose=120):
        """Calculate meal bolus insulin."""
        TDI = self.state["TDI"]
        
        # Correction dose
        correction_factor = 1800 / TDI
        correction_dose = max(0, (glucose - target_glucose) / correction_factor)
        
        # Meal dose
        carb_ratio = 500 / TDI
        meal_dose = (meal_carbs / carb_ratio) * 60  # Convert to U/h
        
        total_bolus = meal_dose + correction_dose
        
        print(f"💉 Bolus Calculation:")
        print(f"   Glucose: {glucose:.1f} mg/dL, Carbs: {meal_carbs:.1f}g")
        print(f"   Meal dose: {meal_dose:.2f} U/h")
        print(f"   Correction: {correction_dose:.2f} U/h") 
        print(f"   Total bolus: {total_bolus:.2f} U/h")
        
        return total_bolus
    
    def run(self, glucose, currentTimeInNs):
        """Main interface method for external system."""
        timestep = int(currentTimeInNs / 10**9)
        
        if timestep > self.sim_time:
            return 0.0
        
        # Handle bolus hold period
        if timestep < self.state["bolusHoldUntil"]:
            insulin_rate = 0.4  # Minimum during bolus hold
            self.state["infusion_values"].append(insulin_rate)
            self.state["basal_rates"].append(insulin_rate)
            self.state["bolus_insulin"].append(0)
            self.state["carb_intake"].append(0)
            print(f"🔒 [Hold] Infusion Rate: {insulin_rate:.2f} U/h")
            return insulin_rate
        
        # Check for meals (20 minutes after meal time)
        meal_carbs = self.meal_schedule.get(timestep - 20, 0)
        give_bolus = meal_carbs > 0
        give_basal = timestep == self.state["next_basal_timestep"] and not give_bolus
        
        print(f"\n⏰ Timestep {timestep}: Meal={meal_carbs:.1f}g, Bolus={give_bolus}, Basal={give_basal}")
        
        if give_bolus:
            # Bolus delivery
            self.state["bolusHoldUntil"] = timestep + 120
            self.state["next_basal_timestep"] = timestep + 120
            self.state["time_since_last_meal"] = 0
            
            total_bolus = self.calculate_bolus(glucose, meal_carbs, timestep)
            
            self.state["bolus_insulin"].append(float(total_bolus))
            self.state["carb_intake"].append(meal_carbs)
            self.state["infusion_values"].append(float(total_bolus))
            
            print(f"💉 Bolus delivery: {total_bolus:.2f} U/h")
            return total_bolus
            
        elif give_basal:
            # RL-tuned basal delivery
            basal_rate = self.predict_insulin_dosage(glucose, timestep)
            self.state["next_basal_timestep"] = timestep + 30
            self.state["time_since_last_insulin"] = 0
            
            self.state["basal_rates"].append(float(basal_rate))
            self.state["infusion_values"].append(float(basal_rate))
            self.state["bolus_insulin"].append(0)
            self.state["carb_intake"].append(0)
            
            print(f"🔧 RL Basal delivery: {basal_rate:.2f} U/h")
            return basal_rate
            
        else:
            # Maintain previous rate
            self.maintain_previous()
            insulin_rate = self.state["infusion_values"][-1]
            print(f"📊 Maintaining rate: {insulin_rate:.2f} U/h")
            return insulin_rate
    
    def maintain_previous(self):
        """Maintain previous basal rate."""
        last_rate = self.state["basal_rates"][-1] if self.state["basal_rates"] else 1.0
        self.state["infusion_values"].append(last_rate)
        self.state["basal_rates"].append(last_rate)
        self.state["bolus_insulin"].append(0)
        self.state["carb_intake"].append(0)
        
        # Update timers
        self.state["time_since_last_meal"] = min(self.state["time_since_last_meal"] + 1, 1440)
        self.state["time_since_last_insulin"] = min(self.state["time_since_last_insulin"] + 1, 120)
    
    def print_summary(self):
        """Print simulation summary."""
        print("\n📊 RL-Tuned PID Diabetes Control Summary:")
        print(f"   Total insulin deliveries: {len(self.state['infusion_values'])}")
        if self.state["infusion_values"]:
            print(f"   Mean insulin rate: {np.mean(self.state['infusion_values']):.2f} U/h")
            print(f"   Total insulin: {sum(self.state['infusion_values']):.2f} U")
        print(f"   TDI limit: {self.state['TDI']:.2f} U/day")
        print(f"   Daily basal limit: {self.state['daily_basal_limit']:.2f} U/day")


# External interface compatibility (matches original artificial_pancreas_simulator.py)
class InsulinSimulator(RLDiabetesSimulator):
    """Compatibility wrapper for external systems."""
    
    def __init__(self, model_path, test_case_path, totalSimulationTimeInNs):
        """Initialize with compatibility for external interface."""
        # Map to our RL model path
        actor_path = os.path.join('A2C', 'save_weights', 'diabetes_actor_best.h5')
        super().__init__(actor_path, test_case_path, totalSimulationTimeInNs)
        print("🔄 Compatibility mode: Using RL-Tuned PID Model")


def main():
    """Test the RL diabetes simulator."""
    print("🩺 Testing RL-Tuned PID Diabetes Simulator")
    print("=" * 60)
    
    # Model and test case paths
    actor_model_path = os.path.join('A2C', 'save_weights', 'diabetes_actor_best.h5')
    test_case_path = os.path.join('..', 'TestData', 'TestCases.txt')
    
    # Simulation time (1 hour = 3600 seconds)
    totalSimulationTimeInNs = 3600 * 10**9
    
    try:
        # Initialize RL simulator
        sim = RLDiabetesSimulator(actor_model_path, test_case_path, totalSimulationTimeInNs)
        
        # Test run
        currentTimeInNs = 1000000000  # Start at 1 second
        
        print(f"\n🔬 Running simulation test...")
        
        for i in range(10):  # Test 10 timesteps
            # Simulate glucose readings
            glucose = np.random.uniform(80, 160)
            
            # Run simulation
            insulin_rate = sim.run(glucose, currentTimeInNs)
            
            print(f"✅ Step {i+1}: Glucose={glucose:.1f} → Insulin={insulin_rate:.2f} U/h")
            
            # Next timestep (1 minute = 60 seconds)
            currentTimeInNs += 60 * 10**9
        
        # Print summary
        sim.print_summary()
        
        print(f"\n🎉 RL-Tuned PID Simulator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()