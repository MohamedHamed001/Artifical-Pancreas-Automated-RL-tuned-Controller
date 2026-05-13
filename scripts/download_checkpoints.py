#!/usr/bin/env python3
"""Thin wrapper around ``python -m ap_rl.scripts.download_checkpoints``.

Lets users run the downloader from a freshly-cloned repo even before
they have done ``pip install -e .``::

    python scripts/download_checkpoints.py --base-url https://...

It simply prepends ``src/`` to ``sys.path`` then delegates to the
package-level CLI.
"""

from __future__ import annotations

import os
import sys


REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def main() -> int:
    from ap_rl.scripts.download_checkpoints import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
