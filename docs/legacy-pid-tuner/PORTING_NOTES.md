# Legacy PID tuner (LunarLander)

This directory contains the original ``Reinforcement_learning_based_PID_Tuner``
upstream project (LunarLander-Continuous demo) that inspired the
PID-tuning RL formulation used in ``ap_rl``. **It is not used by the
diabetes simulator** and is kept here purely for historical reference.

## Notes for anyone trying to run it again

* Hardcoded absolute paths (`/home/diominor/...`) were stripped during the
  refactor. Imports now use ``os.path.dirname(__file__)`` relative
  lookups; saved weights are written to a sibling ``save_weights/``
  folder under each algorithm directory.
* The PPO ``ppo_main.py`` makes ``wandb`` an optional import (the
  upstream copy required it).
* No tests were ported; runtime dependencies include OpenAI Gym (Box2D)
  and Torch, which are **not** in the ``ap-rl`` package's runtime list.

If you want to reuse this implementation, please write new tests and a
proper ``requirements.txt`` first.
