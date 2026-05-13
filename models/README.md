# Models directory

Trained neural-network weights are **not** committed to git. After
`pip install -e .`, download release artifacts into `../checkpoints/`
with:

```bash
ap-rl-download
# or: python scripts/download_checkpoints.py
```

If you vendor local `.h5` exports for demos or CI, you may place copies
here and reference them via `AP_RL_CHECKPOINT_URL` or an absolute
`--dest` path on the download script.
