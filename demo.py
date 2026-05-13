#!/usr/bin/env python3
"""One-command launcher for the Streamlit digital-twin demo.

Equivalent to::

    streamlit run app/app.py

Extra arguments are forwarded to Streamlit, e.g.::

    python demo.py -- --server.port 8502
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    app = repo / "app" / "app.py"
    if not app.is_file():
        raise SystemExit(f"Missing demo entrypoint: {app}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
