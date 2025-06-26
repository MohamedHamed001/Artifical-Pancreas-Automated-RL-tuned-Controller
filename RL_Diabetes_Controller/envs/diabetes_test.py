import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diabetes_pid_env import DiabetesPIDEnv
from A2C.diabetes_a2c_agent import DiabetesA2CAgent

def test_challenging_scenario():
    """Test the diabetes control system with a challenging scenario."""
    
    print("="*60)
    print("DIABETES CONTROL SYSTEM - CHALLENGING SCENARIO TEST")
    print("="*60)
    
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
    
    # Initialize environment
    test_case_file = os.path.join('..', '..', 'TestData', 'TestCases.txt')
    
    env = DiabetesPIDEnv(
        patient_params=patient_parameters,
        test_case_file=test_case_file,
        patient_weight=75,
        target_glucose=120
    )
    
    # Initialize agent
    agent = DiabetesA2CAgent(env)
    
    # Try to load trained weights
    if agent.load_weights('best'):
        print("Loaded trained weights for testing")
        test_mode = "trained"
    else:
        print("No trained weights found. Testing with random policy.")
        test_mode = "random"
    
    print(f"\nTesting with {test_mode} agent...")
    print("Scenario: Multiple meals + exercise + dawn phenomenon")
    
    # Run test episode
    state = env.reset()
    done = False
    step = 0
    
    print("\nStarting simulation...")
    print("Time | BGL   | Insulin | Kp    | Ki    | Kd    | Reward | Event")
    print("-" * 70)
    
    while not done and step < 1440:  # Max 24 hours
        if test_mode == "trained":
            action = agent.actor.get_action(state)
        else:
            # Random policy for comparison
            action = np.random.uniform(-0.1, 0.1, size=3)
        
        next_state, reward, done, info = env.step(action)
        
        # Check for events
        event = ""
        current_meal = env.patient._get_meal_intake(env.patient.time)
        if current_meal > 0:
            event = f"MEAL: {current_meal}g"
        elif env.patient._get_exercise_status(env.patient.time) > 0:
            event = "EXERCISE"
        elif info['bolus_insulin'] > 0:
            event = "BOLUS"
        
        # Print every 30 minutes for readability
        if step % 30 == 0:
            print(f"{step:4d} | {info['glucose']:5.1f} | {info['total_insulin']:7.2f} | "
                  f"{info['Kp']:5.3f} | {info['Ki']:5.3f} | {info['Kd']:5.3f} | "
                  f"{reward:6.1f} | {event}")
        
        state = next_state
        step += 1
        
        # Check for safety violations
        if info['glucose'] < 50:
            print(f"\n*** SEVERE HYPOGLYCEMIA at step {step}! BGL: {info['glucose']:.1f} ***")
            break
        elif info['glucose'] > 250:
            print(f"\n*** SEVERE HYPERGLYCEMIA at step {step}! BGL: {info['glucose']:.1f} ***")
            break
    
    # Get final statistics
    stats = env.get_statistics()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Episode Length: {step} minutes ({step/60:.1f} hours)")
    print(f"Total Reward: {stats['total_episode_reward']:.1f}")
    print(f"Mean Glucose: {stats['mean_glucose']:.1f} mg/dL")
    print(f"Glucose Std: {stats['std_glucose']:.1f} mg/dL")
    print(f"Time in Range (80-140): {stats['time_in_range_80_140']:.1f}%")
    print(f"Time in Range (70-180): {stats['time_in_range_70_180']:.1f}%")
    print(f"Time Hypoglycemic (<70): {stats['time_hypo_70']:.1f}%")
    print(f"Time Hyperglycemic (>180): {stats['time_hyper_180']:.1f}%")
    print(f"Total Insulin Used: {stats['total_insulin']:.1f} U")
    print(f"Mean Insulin Rate: {stats['mean_insulin']:.2f} U/h")
    print(f"Number of Boluses: {stats['num_boluses']}")
    print(f"Final PID Parameters: Kp={stats['final_kp']:.3f}, Ki={stats['final_ki']:.3f}, Kd={stats['final_kd']:.3f}")
    
    # Performance assessment
    print("\n" + "-"*60)
    print("PERFORMANCE ASSESSMENT")
    print("-"*60)
    
    if stats['time_in_range_80_140'] >= 70:
        print("✓ EXCELLENT: >70% time in tight range (80-140 mg/dL)")
    elif stats['time_in_range_70_180'] >= 70:
        print("✓ GOOD: >70% time in acceptable range (70-180 mg/dL)")
    else:
        print("✗ NEEDS IMPROVEMENT: <70% time in range")
    
    if stats['time_hypo_70'] <= 4:
        print("✓ SAFE: <4% time in hypoglycemia")
    else:
        print("✗ UNSAFE: >4% time in hypoglycemia")
    
    if stats['mean_glucose'] >= 80 and stats['mean_glucose'] <= 140:
        print("✓ OPTIMAL: Mean glucose in target range")
    elif stats['mean_glucose'] >= 70 and stats['mean_glucose'] <= 180:
        print("✓ ACCEPTABLE: Mean glucose in safe range")
    else:
        print("✗ SUBOPTIMAL: Mean glucose outside safe range")
    
    # Plot results
    print("\nGenerating plots...")
    env.plot_results()
    
    return stats

def compare_control_methods():
    """Compare different control methods."""
    print("\n" + "="*60)
    print("CONTROL METHODS COMPARISON")
    print("="*60)
    
    methods = ["random", "fixed_pid", "trained"]
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} control...")
        
        # Initialize fresh environment for each test
        patient_parameters = {
            'BW': 75, 'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
            'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
            'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
            'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
            'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30, 'G_init': 10.0,
            'A_EGP': 0.05, 'phi_EGP': -60, 'F_peak': 1.35,
            'K_rise': 5.0, 'K_decay': 0.01, 'G_thresh': 9.0, 'k_R': 0.0031,
        }
        
        test_case_file = os.path.join('..', '..', 'TestData', 'TestCases.txt')
        env = DiabetesPIDEnv(patient_parameters, test_case_file, 75, 120)
        
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < 720:  # 12 hours for comparison
            if method == "random":
                action = np.random.uniform(-0.1, 0.1, size=3)
            elif method == "fixed_pid":
                action = np.array([0.0, 0.0, 0.0])  # No PID changes
            else:  # trained
                agent = DiabetesA2CAgent(env)
                if agent.load_weights('best'):
                    action = agent.actor.get_action(state)
                else:
                    action = np.array([0.0, 0.0, 0.0])
            
            state, reward, done, info = env.step(action)
            step += 1
        
        results[method] = env.get_statistics()
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print("Method        | TIR(80-140) | Mean BGL | Hypo% | Total Reward")
    print("-" * 60)
    
    for method, stats in results.items():
        print(f"{method:12s} | {stats['time_in_range_80_140']:10.1f}% | "
              f"{stats['mean_glucose']:8.1f} | {stats['time_hypo_70']:5.1f}% | "
              f"{stats['total_episode_reward']:12.1f}")
    
    return results

if __name__ == "__main__":
    # Run challenging scenario test
    test_stats = test_challenging_scenario()
    
    # Optionally run comparison
    print("\nWould you like to run a comparison of control methods? (y/n)")
    user_input = input().lower().strip()
    if user_input == 'y':
        comparison_results = compare_control_methods()
    
    print("\nTest completed successfully!") 