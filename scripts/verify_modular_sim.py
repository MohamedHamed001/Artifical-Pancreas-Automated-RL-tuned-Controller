import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.controllers.pid_controller import PIDController
from ap_rl.utils.scenarios import ScenarioLoader
from ap_rl.envs.defaults import DEFAULT_PATIENT_PARAMS as hovorka_params

def run_case_10_modular():
    # 1. Setup Patient Config
    from ap_rl.core.types import PatientConfig

    # We'll use Case 10 body weight (discovered earlier or just use 75.0)
    # Actually let's use the ScenarioLoader to find it if we wanted, but 75kg is fine for demo.
    patient_weight = 75.0
    config = PatientConfig(
        name="Patient_Case10",
        params=hovorka_params,
        body_weight_kg=patient_weight
    )

    # 2. Setup Controller
    # Use PID with standard gains
    controller = PIDController(
        target_mgdl=120.0,
        basal_u_h=1.0, # Estimated
        Kp=0.5, Ki=0.001, Kd=0.05
    )

    # 3. Setup Simulation
    sim_config = SimulationConfig(
        patient_config=config,
        controller=controller,
        duration_min=1440,
        dt_min=5,
        target_glucose_mgdl=120.0,
        basal_rate_u_h=1.0
    )

    runner = SimulationRunner(sim_config)

    # 4. Load Scenario Data
    loader = ScenarioLoader()
    scenario = loader.load_case(10)

    # 5. Run
    print("Starting 24h simulation for Case 10...")
    episode = runner.run(scenario=scenario)
    print(f"Simulation complete. Total Reward: {episode.total_reward:.2f}")

    # 6. Plot Results
    times = [s.time for s in episode.steps]
    glucose = [s.true_glucose for s in episode.steps]
    insulin = [s.delivered_insulin for s in episode.steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(times, glucose, label='Glucose (mg/dL)', color='blue')
    ax1.axhline(120, color='green', linestyle='--', label='Target')
    ax1.axhline(70, color='red', linestyle='--', label='Hypo threshold')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.legend()
    ax1.set_title('Modular Simulator: Case 10 (PID Control)')

    ax2.step(times, insulin, label='Insulin (U/h)', color='orange', where='post')
    ax2.set_ylabel('Insulin Rate (U/h)')
    ax2.set_xlabel('Time (min)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('case10_simulation.png')
    print("Plot saved to case10_simulation.png")

if __name__ == "__main__":
    run_case_10_modular()
