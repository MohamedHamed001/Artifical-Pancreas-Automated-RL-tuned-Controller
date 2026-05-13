# Checkpoints

`.h5` model weights live here at runtime and are **gitignored** (see
the repo-level `.gitignore`).

## Get the latest weights

```bash
export AP_RL_CHECKPOINT_URL=https://github.com/<org>/<repo>/releases/download/v0.1.0
python scripts/download_checkpoints.py
# or, after pip install -e .
ap-rl-download
```

The script downloads:

* `diabetes_actor_best.h5`
* `diabetes_critic_best.h5`

Use `--force` to re-download or `--file NAME` to fetch additional files
(e.g. older episode checkpoints) when present in the Release.

## Producing weights yourself

```bash
ap-rl-train --preset default --seed 42
```

Weights land in this directory automatically; the demo will pick them
up on the next launch.
