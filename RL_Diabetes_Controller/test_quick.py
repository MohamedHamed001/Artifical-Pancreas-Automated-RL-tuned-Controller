#!/usr/bin/env python3
"""
Quick test script to debug the diabetes control components
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.pid_controller import PID
from utils.insulin_calculator import InsulinCalculator

def test_pid_controller():
    """Test PID controller basic functionality"""
    print("=== TESTING PID CONTROLLER ===")
    
    # Create PID controller
    pid = PID(P=0.5, I=0.1, D=0.01)
    pid.SetPoint = 120.0  # Target glucose
    pid.setWindup(50.0)
    
    # Simulate high glucose
    current_glucose = 200.0
    
    print(f"Target: {pid.SetPoint} mg/dL")
    print(f"Current: {current_glucose} mg/dL")
    print(f"Error: {pid.SetPoint - current_glucose} mg/dL")
    
    # Update PID
    pid.update(current_glucose, current_time=1.0)
    
    print(f"PID Output: {pid.output:.3f}")
    print(f"P Term: {pid.PTerm:.3f}")
    print(f"I Term: {pid.ITerm:.3f}")
    print(f"D Term: {pid.DTerm:.3f}")
    
    # Test scaling to insulin rate (with diabetes inversion)
    insulin_rate = np.clip(-pid.output / 10.0, 0.0, 20.0)
    print(f"Scaled Insulin Rate: {insulin_rate:.3f} U/h")
    
    return insulin_rate > 0

def test_insulin_calculator():
    """Test insulin calculator"""
    print("\n=== TESTING INSULIN CALCULATOR ===")
    
    calc = InsulinCalculator(patient_weight_kg=75)
    calc.set_current_time(480)  # 8 AM
    
    print(f"TDI: {calc.tdi:.2f} U")
    print(f"Carb Ratio: {calc.carb_ratio:.2f} g/U")
    print(f"ISF: {calc.isf:.2f} mg/dL/U")
    
    # Test meal bolus
    bolus_result = calc.deliver_bolus(
        carbs_grams=40,
        current_glucose_mgdl=180,
        target_glucose_mgdl=120
    )
    
    print(f"Meal Bolus Result:")
    print(f"  Bolus: {bolus_result['bolus_dose']:.2f} U")
    print(f"  Correction: {bolus_result['correction_dose']:.2f} U")
    print(f"  Total: {bolus_result['total_dose']:.2f} U")
    print(f"  Delivered: {bolus_result['delivered']}")
    
    return bolus_result['delivered'] and bolus_result['total_dose'] > 0

def test_environment_reset():
    """Test environment reset and basic functionality"""
    print("\n=== TESTING ENVIRONMENT ===")
    
    try:
        from envs.diabetes_pid_env import DiabetesPIDEnv
        
        # Patient parameters
        patient_parameters = {
            'BW': 75,
            'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
            'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
            'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
            'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
            'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
            'G_init': 10.0,
            'A_EGP': 0.05, 'phi_EGP': -60,
            'F_peak': 1.35, 'K_rise': 5.0, 'K_decay': 0.01,
            'G_thresh': 9.0, 'k_R': 0.0031,
        }
        
        test_case_file = os.path.join('..', 'TestData', 'TestCases.txt')
        
        env = DiabetesPIDEnv(
            patient_params=patient_parameters,
            test_case_file=test_case_file,
            patient_weight=75,
            target_glucose=120
        )
        
        # Test reset
        state = env.reset()
        print(f"Initial state shape: {state.shape}")
        print(f"Initial PID: Kp={env.pid.Kp:.3f}, Ki={env.pid.Ki:.3f}, Kd={env.pid.Kd:.3f}")
        
        # Test one step with action to increase Kp
        action = np.array([0.1, 0.0, 0.0])  # Increase Kp
        next_state, reward, done, info = env.step(action)
        
        print(f"After step:")
        print(f"  PID: Kp={info['Kp']:.3f}, Ki={info['Ki']:.3f}, Kd={info['Kd']:.3f}")
        print(f"  BGL: {info['glucose']:.1f} mg/dL")
        print(f"  Basal: {info['basal_insulin']:.3f} U/h")
        print(f"  Bolus: {info['bolus_insulin']:.3f} U/h")
        print(f"  Total: {info['total_insulin']:.3f} U/h")
        print(f"  Reward: {reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        return False

if __name__ == "__main__":
    print("QUICK COMPONENT TEST\n")
    
    pid_ok = test_pid_controller()
    insulin_ok = test_insulin_calculator()
    env_ok = test_environment_reset()
    
    print(f"\n=== RESULTS ===")
    print(f"PID Controller: {'✓' if pid_ok else '✗'}")
    print(f"Insulin Calculator: {'✓' if insulin_ok else '✗'}")
    print(f"Environment: {'✓' if env_ok else '✗'}")
    
    if all([pid_ok, insulin_ok, env_ok]):
        print("All components working! ✓")
    else:
        print("Some components need fixing! ✗") 