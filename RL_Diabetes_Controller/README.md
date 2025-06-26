# RL-Based Adaptive PID Diabetes Controller

This project implements a Reinforcement Learning-based adaptive PID controller for Type 1 Diabetes insulin delivery. The controller dynamically tunes PID parameters to maintain glucose levels within target ranges while handling meals, exercise, and circadian variations.

## Key Features

- **Adaptive PID Control**: RL agent learns to adjust Kp, Ki, Kd parameters based on patient state
- **Insulin Delivery System**: 
  - Basal insulin via RL-tuned PID
  - Automated meal bolus calculation
  - Correction dose calculation
  - 15-minute safety lockout
- **Safety Constraints**: 
  - Episode termination for dangerous glucose levels
  - Progressive penalty system
- **Hovorka Patient Model**: Advanced diabetes simulation environment

## Architecture

```
RL Agent → PID Parameter Tuning → PID Controller → Basal Insulin
    ↑                                     ↓
Patient State ← Hovorka Model ← Total Insulin (Basal + Bolus + Correction)
```

## Reward System

- **Violation (Episode End)**: BGL < 50 or > 250 mg/dL
- **Excellent (+15)**: 80-140 mg/dL  
- **Good (+5)**: 70-79, 141-180 mg/dL
- **Penalized (-10)**: BGL < 70 or > 180 mg/dL
- **Additional Penalties**: Rapid insulin changes, glucose rate of change

## Usage

### Training
```bash
cd ./A2C/
python diabetes_a2c_main.py
```

### Testing
```bash
cd ./envs/
python diabetes_test.py
```

## Requirements
```
tensorflow==2.5.0
scikit-learn==0.23.2
matplotlib==3.8.3
gym
numpy
scipy
pandas
```

## Project Structure
```
RL_Diabetes_Controller/
├── A2C/
│   ├── diabetes_a2c_agent.py    # Main RL agent
│   ├── diabetes_a2c_actor.py    # Actor network
│   ├── diabetes_a2c_critic.py   # Critic network
│   ├── diabetes_a2c_main.py     # Training script
│   └── save_weights/             # Model checkpoints
├── envs/
│   ├── diabetes_pid_env.py      # Main environment
│   ├── diabetes_test.py         # Testing script
│   └── challenging_scenarios/   # Test scenarios
├── data/
│   └── test_cases/              # Meal and exercise data
└── utils/
    ├── insulin_calculator.py    # Bolus/correction calculations
    └── meal_parser.py          # Test case parser
``` 