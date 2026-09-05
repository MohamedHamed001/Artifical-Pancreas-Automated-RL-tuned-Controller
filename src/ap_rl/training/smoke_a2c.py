import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ap_rl.envs import DiabetesPIDEnv
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS
from ap_rl.agents import DiabetesA2CAgent

def smoke_test():
    print("Initializing environment...")
    env = DiabetesPIDEnv(
        patient_params=dict(DEFAULT_PATIENT_PARAMS),
        test_case_id=1
    )

    print("Initializing agent...")
    agent = DiabetesA2CAgent(env)

    print("Running 1 episode smoke test...")
    agent.train(max_episodes=1, verbose=True)

    print("\nSmoke test passed!")

if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        print(f"\nSmoke test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
