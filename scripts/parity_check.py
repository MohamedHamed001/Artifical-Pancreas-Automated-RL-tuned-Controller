import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.controllers.pid_controller import PIDController
from ap_rl.core.types import PatientConfig
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS

from ap_rl.controllers.base import Controller

class OpenLoopController(Controller):
    def __init__(self, insulin_sequence):
        self.insulin = insulin_sequence
        self.idx = 0
    def get_action(self, state, info=None):
        if self.idx < len(self.insulin):
            val = self.insulin[self.idx]
            self.idx += 1
            return np.array([val], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
    def reset(self): self.idx = 0
    def update(self, reward, done): pass

def run_parity_check():
    # 1. Load Baseline
    with open("baseline_capture.json", "r") as f:
        baseline = json.load(f)

    # 2. Setup Patient Config (matching baseline metadata)
    g_init_mgdl = baseline["glucose"][0]
    patient_params = DEFAULT_PATIENT_PARAMS.copy()
    patient_params["G_init"] = g_init_mgdl

    patient_config = PatientConfig(
        name="Parity_Patient",
        params=patient_params,
        body_weight_kg=75.0
    )

    # 3. Setup Open Loop Controller to replay exact insulin
    controller = OpenLoopController(baseline["insulin"])

    # 4. Setup Simulation
    sim_config = SimulationConfig(
        patient_config=patient_config,
        controller=controller,
        duration_min=len(baseline["times"]),
        dt_min=1,
        target_glucose_mgdl=120.0,
        basal_rate_u_h=1.0
    )

    runner = SimulationRunner(sim_config)

    # 5. Run (Inject meals from baseline)
    print(f"Running parity simulation for {sim_config.duration_min} minutes (Open Loop)...")
    episode = runner.run(meal_data=baseline.get("meals", []))

    # 6. Compare
    sim_glucose = np.array([s.true_glucose for s in episode.steps])
    base_glucose = np.array(baseline["glucose"])

    # Trim to same length
    min_len = min(len(sim_glucose), len(base_glucose))
    sim_glucose = sim_glucose[:min_len]
    base_glucose = base_glucose[:min_len]

    print(f"Initial Glucose (Baseline): {base_glucose[0]:.4f}")
    print(f"Initial Glucose (Simulator): {sim_glucose[0]:.4f}")
    print(f"First 5 steps (Baseline): {base_glucose[:5]}")
    print(f"First 5 steps (Simulator): {sim_glucose[:5]}")

    mse = np.mean((sim_glucose - base_glucose)**2)
    max_err = np.max(np.abs(sim_glucose - base_glucose))

    print(f"Parity Results:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Max Absolute Error: {max_err:.6f}")

    # 7. Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(base_glucose, 'r--', label='Legacy Baseline', alpha=0.7)
    plt.plot(sim_glucose, 'b-', label='Modular Simulator', alpha=0.7)
    plt.title('Physics Engine Parity Check')
    plt.xlabel('Time (min)')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.grid(True)
    plt.savefig('parity_comparison.png')
    print("Plot saved to parity_comparison.png")

    if max_err < 1.0: # Tolerance of 1 mg/dL for now
        print("SUCCESS: Numerical parity achieved!")
    else:
        print("WARNING: Significant drift detected. Check controller implementation or dt_min resolution.")

if __name__ == "__main__":
    run_parity_check()
