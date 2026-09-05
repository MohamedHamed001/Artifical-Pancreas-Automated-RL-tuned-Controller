from ap_rl.core.types import PatientConfig
from ap_rl.simulation.simulator import SimulationRunner, SimulationConfig
from ap_rl.controllers.mpc_controller import GlucoseMPC
from ap_rl.controllers.pid_controller import PIDController

def run_sanity_check():
    # 1. Setup Patient
    config = PatientConfig(
        name="SanityPatient",
        params={"BW": 75.0, "G_init": 140.0}, # Start slightly high
        body_weight_kg=75.0
    )

    # 2. Setup Controller (MPC)
    mpc = GlucoseMPC(target_mgdl=110.0, basal_u_h=1.2, isf=50.0)

    # 3. Setup Simulation
    sim_config = SimulationConfig(
        patient_config=config,
        controller=mpc,
        duration_min=300, # 5 hours
        dt_min=5,
        target_glucose_mgdl=110.0,
        basal_rate_u_h=1.2
    )

    runner = SimulationRunner(sim_config)

    # 4. Run (with a small meal at t=60)
    meals = [{"time": 60, "carbs": 40.0}]
    record = runner.run(meal_data=meals)

    print(f"Simulation completed for {record.patient_id} using {record.controller_name}")
    print(f"Total Steps: {len(record.steps)}")
    print(f"Initial Glucose: {record.steps[0].true_glucose:.1f} mg/dL")
    print(f"Final Glucose: {record.steps[-1].true_glucose:.1f} mg/dL")
    print(f"Total Reward: {record.total_reward:.1f}")

    # Simple check: BGL should eventually trend toward target if MPC works
    # (even if it takes longer than 5 hours)

    # Check for any safety events
    safety_events = [s.safety_events for s in record.steps if s.safety_events]
    if safety_events:
        print(f"Safety events triggered: {safety_events}")

if __name__ == "__main__":
    run_sanity_check()
