import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from diabetes_a2c_agent import DiabetesA2CAgent
from envs.diabetes_pid_env import DiabetesPIDEnv

def main():
    """Main training function for diabetes control."""
    
    # Patient parameters (Hovorka model)
    patient_parameters = {
        'BW': 75,  # Body weight in kg
        # Hovorka model parameters
        'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
        'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
        'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
        'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
        'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
        'G_init': 10.0,  # Initial glucose 180 mg/dL
        
        # Custom parameters
        'A_EGP': 0.05,    # Circadian variation
        'phi_EGP': -60,   # Phase shift
        'F_peak': 1.35,   # Exercise sensitivity
        'K_rise': 5.0,
        'K_decay': 0.01,
        'G_thresh': 9.0,  # Renal threshold
        'k_R': 0.0031,    # Renal clearance
    }
    
    # Initialize environment
    test_case_file = os.path.join('..', 'TestData', 'TestCases.txt')
    
    print("Initializing Diabetes PID Environment...")
    env = DiabetesPIDEnv(
        patient_params=patient_parameters,
        test_case_file=test_case_file,
        patient_weight=75,
        target_glucose=120
    )
    
    # Initialize A2C agent
    print("Initializing A2C Agent...")
    agent = DiabetesA2CAgent(env)
    
    # Training configuration
    MAX_EPISODES = 500  # Number of training episodes
    SAVE_FREQUENCY = 25  # Save weights every N episodes
    
    print(f"\n{'='*50}")
    print("DIABETES CONTROL TRAINING")
    print(f"{'='*50}")
    print(f"Episodes: {MAX_EPISODES}")
    print(f"Environment: Hovorka Patient Model")
    print(f"Control: RL-tuned PID + Meal Bolus + Correction")
    print(f"Target: Maintain glucose 80-140 mg/dL")
    print(f"{'='*50}\n")
    
    try:
        # Train the agent
        agent.train(
            max_episodes=MAX_EPISODES,
            plot_progress=False,  # Disabled for unattended training
            save_frequency=SAVE_FREQUENCY,
            verbose=True
        )
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_rewards, test_stats = agent.test(
            num_episodes=3,
            render=True,
            load_best=True
        )
        
        print(f"\nTest Results Summary:")
        for i, (reward, stats) in enumerate(zip(test_rewards, test_stats)):
            print(f"  Test {i+1}: Reward={reward:.1f}, "
                  f"TIR(80-140)={stats['time_in_range_80_140']:.1f}%, "
                  f"Mean BGL={stats['mean_glucose']:.1f}")
        
        print(f"\n{'='*50}")
        print("ALL DONE! Check the plots and saved weights.")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current progress...")
        agent.save_weights('interrupted')
        print("Progress saved. You can resume training later.")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Saving current progress...")
        agent.save_weights('error_save')
        raise

if __name__ == "__main__":
    main() 