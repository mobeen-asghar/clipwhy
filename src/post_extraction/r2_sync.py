"""Read-only R2 sync.

Pulls features/, clip_embeddings/, and pairs/ to a local cache directory.
Never writes to R2. We treat the bucket as immutable for this phase of work.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

log = logging.getLogger("clipwhy.post_extraction.r2_sync")


def _rclone(*args: str) -> int:
    if shutil.which("rclone") is None:
        log.error("rclone not found on PATH")
        return 127
    cmd = ["rclone", *args, "--disable-http2"]
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def pull_features() -> int:
    return _rclone(
        "copy", config.R2_FEATURES_PREFIX, str(config.R2_FEATURES_DIR),
        "--transfers", "16", "--checkers", "16",
        "--update", "--progress",
    )


def pull_clip_embeddings() -> int:
    return _rclone(
        "copy", config.R2_CLIP_EMBEDDINGS_PREFIX, str(config.R2_CLIP_EMBEDDINGS_DIR),
        "--transfers", "8", "--checkers", "16",
        "--update", "--progress",
    )


def pull_pairs() -> int:
    pairs_dir = config.R2_CACHE / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    return _rclone(
        "copy", f"{config.R2_REMOTE}/pairs/", str(pairs_dir),
        "--transfers", "16", "--checkers", "16",
        "--update", "--progress",
    )


def pull_all() -> int:
    config.ensure_dirs()
    for name, fn in [
        ("pairs", pull_pairs),
        ("features", pull_features),
        ("clip_embeddings", pull_clip_embeddings),
    ]:
        log.info("=== Pulling %s ===", name)
        rc = fn()
        if rc != 0:
            log.error("%s pull failed with rc=%d", name, rc)
            return rc
    log.info("=== All pulls complete ===")
    return 0


def verify_local() -> dict:
    """Quick count of local cache contents."""
    counts = {
        "features": sum(1 for _ in config.R2_FEATURES_DIR.glob("CR*_features.csv")),
        "clip_embeddings": sum(1 for _ in config.R2_CLIP_EMBEDDINGS_DIR.glob("CR*_clip_embeddings.npz")),
        "pairs": sum(1 for _ in (config.R2_CACHE / "pairs").glob("CR*_whisper_pairs.csv")),
    }
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(pull_all())
