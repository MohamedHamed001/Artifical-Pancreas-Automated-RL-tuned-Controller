"""Keras 3+ requires ``save_weights`` paths to end in ``.weights.h5``.

We standardise on that suffix for new checkpoints while still discovering
legacy ``*.h5`` files when loading or downloading from older releases.
"""

from __future__ import annotations

import os

# Canonical filenames under checkpoints/
ACTOR_BEST = "diabetes_actor_best.weights.h5"
CRITIC_BEST = "diabetes_critic_best.weights.h5"

DEFAULT_BUNDLE: tuple[str, ...] = (ACTOR_BEST, CRITIC_BEST)


def actor_critic_paths(save_dir: str | os.PathLike, name: str) -> tuple[str, str]:
    """Return (actor_path, critic_path) for a given checkpoint stem ``name``."""
    root = os.fspath(save_dir)
    return (
        os.path.join(root, f"diabetes_actor_{name}.weights.h5"),
        os.path.join(root, f"diabetes_critic_{name}.weights.h5"),
    )


def actor_load_candidates(actor_filename: str, ckpt_dir: str | os.PathLike) -> list[str]:
    """Paths to try when loading an actor (new suffix first, then legacy)."""
    d = os.fspath(ckpt_dir)
    stem = actor_filename
    if stem.endswith(".weights.h5"):
        legacy = stem.replace(".weights.h5", ".h5")
        return [os.path.join(d, stem), os.path.join(d, legacy)]
    if stem.endswith(".h5") and not stem.endswith(".weights.h5"):
        modern = stem.replace(".h5", ".weights.h5")
        return [os.path.join(d, modern), os.path.join(d, stem)]
    return [os.path.join(d, stem)]


def remote_urls_for_file(base_url: str, local_filename: str) -> list[str]:
    """URLs to try when downloading (prefer modern name, then GitHub legacy)."""
    base = base_url.rstrip("/")
    urls = [f"{base}/{local_filename}"]
    if local_filename.endswith(".weights.h5"):
        legacy = local_filename.replace(".weights.h5", ".h5")
        urls.append(f"{base}/{legacy}")
    return urls
