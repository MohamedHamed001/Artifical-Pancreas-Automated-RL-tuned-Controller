# Legacy RL–PID tuner (LunarLander)

The directory [`legacy-pid-tuner/`](legacy-pid-tuner/) archives the original
**LunarLander**-based reinforcement-learning PID tuning codebase that inspired
the diabetes-specific implementation in `src/ap_rl/`.

- **Not used** by the artificial pancreas demo or training CLI.
- **Hardcoded machine paths** in the original upstream tree were removed or
  replaced with comments when this snapshot was archived.
- For diabetes control, use `ap_rl.envs.DiabetesPIDEnv` and
  `ap_rl.training.train_a2c` instead.

See [`legacy-pid-tuner/PORTING_NOTES.md`](legacy-pid-tuner/PORTING_NOTES.md)
for import and dependency notes if you revive the LunarLander experiments.
