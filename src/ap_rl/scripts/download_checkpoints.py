"""Download trained A2C checkpoints from a GitHub Release (or any HTTPS URL).

Usage (after ``pip install -e .``)::

    ap-rl-download                             # uses AP_RL_CHECKPOINT_URL env var
    ap-rl-download --base-url https://...      # explicit base URL

Or directly::

    python -m ap_rl.scripts.download_checkpoints

Notes:
    Local files use the ``*.weights.h5`` suffix required by Keras 3
    ``Model.save_weights``. If a release only hosts legacy ``*.h5``
    assets, this script tries those URLs and still saves under the
    modern filenames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from ap_rl.utils.checkpoint_filenames import DEFAULT_BUNDLE, remote_urls_for_file
from ap_rl.utils.paths import checkpoints_dir

# Placeholder; users override via env or --base-url.
PLACEHOLDER_URL = (
    "https://github.com/<your-org>/Artifical-Pancreas-Automated-RL-tuned-Controller"
    "/releases/download/v0.1.0"
)


def _resolve_base_url(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value.rstrip("/")
    env_value = os.environ.get("AP_RL_CHECKPOINT_URL", "").strip()
    if env_value:
        return env_value.rstrip("/")
    raise SystemExit(
        "ERROR: no checkpoint URL configured.\n"
        "Set AP_RL_CHECKPOINT_URL (see .env.example) or pass --base-url.\n"
        f"Example: --base-url {PLACEHOLDER_URL}"
    )


def _download_one_with_fallbacks(base: str, local_filename: str, dest: Path) -> None:
    """Stream ``local_filename`` from ``base`` into ``dest`` (try legacy ``.h5``)."""
    import requests

    errors: list[str] = []
    for url in remote_urls_for_file(base, local_filename):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    errors.append(f"{url} (HTTP {response.status_code})")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            print(f"  ✓ saved {dest.name} ({dest.stat().st_size / 1024:.1f} KiB) ← {url}")
            return
        except OSError as exc:
            errors.append(f"{url} ({exc})")
        except Exception as exc:  # pragma: no cover - requests errors
            errors.append(f"{url} ({exc})")
    raise RuntimeError(
        "could not download "
        + local_filename
        + " — tried:\n  - "
        + "\n  - ".join(errors)
    )


def download(
    base_url: Optional[str] = None,
    files: Iterable[str] = DEFAULT_BUNDLE,
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> int:
    """Download ``files`` from ``base_url`` into ``dest_dir``.

    Returns 0 on success, non-zero on failure.
    """
    base = _resolve_base_url(base_url)
    target = Path(dest_dir) if dest_dir is not None else checkpoints_dir()
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoints from {base}")
    print(f"Target directory: {target}")
    for filename in files:
        dest = target / filename
        if dest.exists() and not force:
            print(f"  - {filename} already exists, skipping (use --force to overwrite)")
            continue
        try:
            _download_one_with_fallbacks(base, filename, dest)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # pragma: no cover - network errors
            print(f"  ! failed to download {filename}: {exc}", file=sys.stderr)
            return 1
    print("Done.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download A2C actor/critic weights (*.weights.h5) into checkpoints/.",
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Base URL hosting the weight files. Defaults to the AP_RL_CHECKPOINT_URL "
            "environment variable."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (defaults to <repo>/checkpoints/).",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Filename(s) to download. Repeat for multiple files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a file already exists.",
    )
    args = parser.parse_args(argv)

    files = tuple(args.files) if args.files else DEFAULT_BUNDLE
    return download(
        base_url=args.base_url,
        files=files,
        dest_dir=args.dest,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
