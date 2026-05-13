"""ap_rl: Artificial Pancreas RL-tuned PID controller package.

Top-level modules:
- ``ap_rl.envs``: Hovorka virtual-patient ODE + PID-tuning environment.
- ``ap_rl.agents``: A2C actor/critic networks and orchestrating agent.
- ``ap_rl.utils``: PID, insulin math, meal parsing, paths, seeding, configs.
- ``ap_rl.visualization``: Publication-quality matplotlib helpers.
- ``ap_rl.training``: Consolidated A2C training entry point.
- ``ap_rl.runtime``: Thin rollout helpers shared by demo and tests.
- ``ap_rl.scripts``: CLI utilities (checkpoint downloader, ...).

This package is intentionally **decoupled from any UI**. The Streamlit
demo in ``app/app.py`` is an orchestration layer only.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
