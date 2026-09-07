# Phase 0C Reproducible Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible Python 3.11 dependency and container baseline with `uv`.

**Architecture:** `pyproject.toml` remains the dependency-intent source, while `uv.lock` freezes the complete graph. Optional dependency groups reflect real subsystem boundaries, and Docker consumes the same lock rather than resolving a separate graph.

**Tech Stack:** Python 3.11, PEP 621/setuptools, uv 0.11.x, Docker, pytest

**Spec:** `docs/superpowers/specs/2026-09-05-phase-0-runtime-design.md`

## Global Constraints

- Support Python `>=3.11,<3.12` during stabilization.
- Keep Gymnasium in base dependencies.
- Put TensorFlow and h5py in `rl`, CVXPY in `mpc`, Streamlit and Plotly in `demo`, Numba in `fast`, and pytest plus pytest-mock in `dev`.
- Remove hand-maintained `requirements.txt`; do not replace it with another manually maintained dependency list.
- Commit `uv.lock` and `.python-version`; `.python-version` contains `3.11`.
- Docker installs from `uv.lock` and retains the temporary Streamlit command.
- Do not alter simulation, controller, reward, training, or frontend behavior.
- Do not commit or push. The user must explicitly authorize those operations in a later prompt.

---

### Task 1: Reproducible Python and dependency baseline

**Files:**
- Create: `.python-version`
- Create: `.dockerignore`
- Create: `uv.lock`
- Modify: `pyproject.toml`
- Modify: `Dockerfile`
- Modify: `README.md`
- Modify: `docs/phase-0-baseline.md`
- Delete: `requirements.txt`

**Interfaces:**
- Consumes: the dependency boundaries and Python policy in the linked design.
- Produces: `uv sync --locked --all-extras` as the contributor setup contract and a Docker build resolved from the same lockfile.

- [ ] **Step 1: Update package metadata**

  Set `requires-python = ">=3.11,<3.12"`, retain only the Python 3.11 classifier,
  keep the base dependencies listed in the spec, and create the exact `rl`,
  `mpc`, `demo`, `dev`, and `fast` groups. Add `h5py>=3.8` to `rl`. Delete the
  redundant `gym` group and delete `requirements.txt`.

- [ ] **Step 2: Select Python and generate the lock**

  Create `.python-version` containing `3.11`, run `uv lock`, and run
  `uv lock --check`. Do not hand-edit `uv.lock`.

- [ ] **Step 3: Make Docker consume the lock**

  Keep `python:3.11-slim` and the temporary Streamlit command. Copy a pinned uv
  binary from `ghcr.io/astral-sh/uv:0.11.14`, copy `pyproject.toml`, `uv.lock`,
  and `README.md`, then run
  `uv sync --frozen --no-install-project --no-dev --extra demo --extra mpc`.
  Copy the source/application/config/data/script/model directories and run
  `uv sync --frozen --no-dev --extra demo --extra mpc` again to install the
  project. Put `/app/.venv/bin` on `PATH`. Create `.dockerignore` with explicit
  entries for `.git`, `.venv`, Python/test/tool caches, checkpoints, `scratch`,
  and the generated root artifacts already named in `.gitignore`.

- [ ] **Step 4: Update contributor documentation**

  Replace pip/venv setup instructions with `uv sync --locked --all-extras` and
  `uv run` commands. Preserve the warning that the branch is WIP and Streamlit
  is deprecated. Update `docs/phase-0-baseline.md` so its verified commands use
  the locked environment and explain the dependency groups.

- [ ] **Step 5: Verify in a fresh isolated environment**

  Use a temporary directory through `UV_PROJECT_ENVIRONMENT` so verification
  does not rely on the broken repository `.venv`. Run this exact sequence:

  ```bash
  uv lock --check
  task_uv_env_dir=$(/usr/bin/mktemp -d)
  UV_PROJECT_ENVIRONMENT="$task_uv_env_dir/venv" uv sync --locked --all-extras
  uv pip check --python "$task_uv_env_dir/venv/bin/python"
  "$task_uv_env_dir/venv/bin/python" -m pytest -p no:cacheprovider -q
  "$task_uv_env_dir/venv/bin/python" scripts/smoke_baseline_rollout.py
  "$task_uv_env_dir/venv/bin/python" scripts/verify_modular_sim.py
  PYTHONPATH=src "$task_uv_env_dir/venv/bin/python" -m ap_rl.scripts.sanity_check_sim
  ```

  Run `docker build -t ap-rl-phase0 .` only if `docker info` confirms an
  available daemon. Report an unavailable daemon explicitly.

- [ ] **Step 6: Self-review without committing**

  Confirm `git diff --check` is clean, dependency names appear in exactly one
  hand-maintained manifest, no generated artifacts are tracked, and no source
  behavior changed. Write the implementation and test evidence to the assigned
  report file. Do not commit or push.
