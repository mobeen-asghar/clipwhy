"""
Progress tracking for the per-creator pipeline.

Each creator gets a JSON marker file on shared storage:
  - {creator_id}_done.json   = completed (or skipped)
  - {creator_id}_error.json  = failed, retryable on next run

On restart, done creators are skipped. Errored creators are retried.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("clipwhy.progress")


def _progress_dir() -> Path:
    config.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return config.PROGRESS_DIR


def is_done(creator_id: str) -> bool:
    """Check if a creator has already been fully processed (or skipped)."""
    return (_progress_dir() / f"{creator_id}_done.json").exists()


def mark_done(creator_id: str, metadata: dict):
    """Mark a creator as fully processed."""
    metadata["status"] = metadata.get("status", "done")
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    path = _progress_dir() / f"{creator_id}_done.json"
    path.write_text(json.dumps(metadata, indent=2))
    # Remove any previous error marker
    err_path = _progress_dir() / f"{creator_id}_error.json"
    if err_path.exists():
        err_path.unlink()
    log.info("%s marked done", creator_id)


def mark_skipped(creator_id: str, reason: str):
    """Mark a creator as skipped (counts as done, won't retry)."""
    mark_done(creator_id, {"status": "skipped", "reason": reason})
    log.info("%s skipped: %s", creator_id, reason)


def mark_error(creator_id: str, error: str):
    """Mark a creator as failed (retryable on next run)."""
    path = _progress_dir() / f"{creator_id}_error.json"
    path.write_text(json.dumps({
        "status": "error",
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    log.error("%s errored: %s", creator_id, error)


def get_status(creator_id: str) -> str:
    """Return 'done', 'error', or 'pending' for a creator."""
    if (_progress_dir() / f"{creator_id}_done.json").exists():
        return "done"
    if (_progress_dir() / f"{creator_id}_error.json").exists():
        return "error"
    return "pending"


def load_done_metadata(creator_id: str) -> dict:
    """Load the done marker metadata for a completed creator."""
    path = _progress_dir() / f"{creator_id}_done.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ── Per-step caching within a creator ───────────────────────────────────────
# These caches let a worker resume mid-creator after a crash.


def save_step_cache(creator_id: str, step_name: str, data: dict):
    """Save intermediate results for a specific step."""
    path = _progress_dir() / f"{creator_id}_{step_name}.json"
    path.write_text(json.dumps(data, default=str))


def load_step_cache(creator_id: str, step_name: str) -> dict | None:
    """Load cached results for a step, or None if not cached."""
    path = _progress_dir() / f"{creator_id}_{step_name}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def clear_step_caches(creator_id: str):
    """Remove all step caches for a creator (called after mark_done)."""
    for path in _progress_dir().glob(f"{creator_id}_step*.json"):
        path.unlink(missing_ok=True)


# ── Assignment ──────────────────────────────────────────────────────────────


def load_assignment(vm_id: str) -> list[str]:
    """Load the creator IDs assigned to this VM."""
    path = config.INPUT_DIR / "assignment.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {path}. "
            "Run 'cli.py assign' first to generate it."
        )
    assignments = json.loads(path.read_text())
    if vm_id not in assignments:
        raise KeyError(f"VM '{vm_id}' not found in assignment. Available: {list(assignments.keys())}")
    return assignments[vm_id]


def save_assignment(assignments: dict):
    """Save the VM-to-creator assignment mapping."""
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.INPUT_DIR / "assignment.json"
    path.write_text(json.dumps(assignments, indent=2))
    log.info("Assignment saved: %s", {k: len(v) for k, v in assignments.items()})


# ── Summary stats ───────────────────────────────────────────────────────────


def get_progress_summary(creator_ids: list[str]) -> dict:
    """Get overall progress counts for a list of creators."""
    done = 0
    skipped = 0
    errors = 0
    pending = 0
    total_segments = 0
    total_positives = 0
    total_pairs = 0
    total_caption_pairs = 0
    total_dropped = 0
    cpu_done = 0

    for cid in creator_ids:
        status = get_status(cid)
        if status == "done":
            meta = load_done_metadata(cid)
            if meta.get("status") == "skipped":
                skipped += 1
            elif meta.get("status") == "cpu_done":
                cpu_done += 1
                done += 1
                total_segments += meta.get("total_segments", 0)
                total_pairs += meta.get("pairs_found", 0)
            else:
                done += 1
                total_segments += meta.get("total_segments", 0)
                total_positives += meta.get("positive_segments", 0)
                total_pairs += meta.get("pairs_found", 0)
                total_caption_pairs += meta.get("caption_pairs", 0)
                total_dropped += meta.get("dropped_pairs", 0)
        elif status == "error":
            errors += 1
        else:
            pending += 1

    return {
        "done": done,
        "skipped": skipped,
        "cpu_done": cpu_done,
        "errors": errors,
        "pending": pending,
        "total": len(creator_ids),
        "total_segments": total_segments,
        "total_positives": total_positives,
        "total_pairs": total_pairs,
        "total_caption_pairs": total_caption_pairs,
        "total_dropped": total_dropped,
        "positive_rate": (
            f"{total_positives / total_segments * 100:.1f}%"
            if total_segments > 0 else "N/A"
        ),
    }
