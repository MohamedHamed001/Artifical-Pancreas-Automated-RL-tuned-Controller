import os
import sys

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore

# Ensure the legacy tuner's package layout is importable when this script
# is run directly (e.g. ``python ppo_main.py`` from this folder).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.sample_env import PIDsampleEnv
from ppo_agent import PPOTunner

def main():
    max_episode_num = 100000
    env = PIDsampleEnv(set_point=1)
    tunner = PPOTunner(env)
    tunner.train(max_episode_num, plot=1, on_wandb=True)

if __name__ =='__main__':
    main()