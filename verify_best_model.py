import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append("src")

from ap_rl.training.train_a2c import build_env
from ap_rl.agents.diabetes_a2c_agent import DiabetesA2CAgent

def verify():
    # Use Case 16 (Heavy Day with large meals)
    case_id = 16
    print(f"Loading Case {case_id} (Heavy Day Evaluation)...")
    env = build_env(seed=None, case_id=case_id)
    
    # Initialize agent
    agent = DiabetesA2CAgent(env)
    
    # Load best weights from models/
    print("Loading 'best' weights from models/...")
    success = agent.load_weights("best")
    if not success:
        print("Failed to load 'best' weights. Checking 'final'...")
        success = agent.load_weights("final")
    
    if not success:
        print("Could not find any saved weights.")
        return

    # Run one episode
    print("Running evaluation episode...")
    state = env.reset()
    done = False
    
    history = {
        "glucose": [],
        "kp": [],
        "ki": [],
        "kd": [],
        "reward": [],
        "insulin": []
    }
    
    while not done:
        action = agent.actor.get_action(state)
        state, reward, done, info = env.step(action)
        
        history["glucose"].append(info["glucose"])
        history["kp"].append(info["Kp"])
        history["ki"].append(info["Ki"])
        history["kd"].append(info["Kd"])
        history["reward"].append(reward)
        history["insulin"].append(info["total_insulin"])
    
    stats = env.get_statistics()
    print("\n--- Results ---")
    print(f"TIR 70-180: {stats['time_in_range_70_180']:.1f}%")
    print(f"TIR 80-140: {stats['time_in_range_80_140']:.1f}%")
    print(f"Mean Glucose: {stats['mean_glucose']:.1f} mg/dL")
    print(f"Max BGL: {max(history['glucose']):.1f} mg/dL")
    print(f"Min BGL: {min(history['glucose']):.1f} mg/dL")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # BGL Plot
    ax1.plot(history["glucose"], label="BGL", color="black", linewidth=2)
    ax1.axhline(120, color="green", linestyle="--", alpha=0.5, label="Target")
    ax1.axhline(180, color="red", linestyle="--", alpha=0.3)
    ax1.axhline(70, color="red", linestyle="--", alpha=0.3)
    ax1.fill_between(range(len(history["glucose"])), 70, 180, color="green", alpha=0.1, label="Target Range")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title(f"A2C Verification - Case {case_id} (Heavy Day)")
    ax1.legend()
    
    # PID Gains Plot
    ax2.plot(history["kp"], label="Kp", color="blue")
    ax2.plot(history["ki"], label="Ki * 10", color="orange") # Scale Ki for visibility
    ax2.set_ylabel("PID Gains")
    ax2.set_xlabel("Time (min)")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = "verification_result.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    verify()
