"""Pod-level orchestrator with prefetch.

Two-thread pod-level architecture:

  ┌─────────────────────────────────────────────────────────────┐
  │  Main thread                     Prefetch thread            │
  │  ────────────                    ────────────────           │
  │  (blocks until                   claim_next()               │
  │   prefetch puts                  pull_inputs()              │
  │   a creator)                     put on queue (maxsize=1)   │
  │       ↓                              ↓                      │
  │  process_creator(skip_pull=True)  (blocks until main takes) │
  │  cleanup_local()                  loop: claim next          │
  │  empty_cache()                                              │
  │       ↓                                                     │
  │  (loop to pull next from queue)                             │
  └─────────────────────────────────────────────────────────────┘

Claim ownership: both threads can hold a claim at the same time. We track
both so SIGTERM releases both.

Disk pressure: at any moment up to 2 creators' media is on disk (the one
being processed + the one being prefetched). For ~3 GB per creator +
~3 GB of models, peak disk is ~11-12 GB, well under the 30 GB volume.
"""
import logging
import os
import queue
import signal
import sys
import threading
import time

import requests

from . import claims, config
from .worker import _pull_inputs, cleanup_local, process_creator

log = logging.getLogger("clipwhy.features.orchestrator")

# Shutdown state. Updated from signal handler + orchestrator threads.
_shutdown_flag = {
    "stop": False,
    "current_creator": None,
    "prefetch_creator": None,
    "vm_id": None,
}


def _install_signal_handlers():
    def handler(signum, frame):
        log.warning("Received signal %d, graceful shutdown requested", signum)
        _shutdown_flag["stop"] = True
        vm = _shutdown_flag["vm_id"]
        for key in ("current_creator", "prefetch_creator"):
            cid = _shutdown_flag.get(key)
            if cid and vm:
                try:
                    claims.release(cid, vm)
                    log.warning("Released claim on %s (%s)", cid, key)
                except Exception as e:
                    log.error("Release on shutdown failed for %s: %s", cid, e)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _discord(msg: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    try:
        requests.post(url, json={"content": msg[:1900]}, timeout=5)
    except Exception:
        pass


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Sentinels for the prefetch queue.
_POOL_EMPTY = "__pool_empty__"


def _prefetch_loop(vm_id: str, q: "queue.Queue", stop: threading.Event) -> None:
    """Runs on a background thread. Claims + pulls next creator ahead of time."""
    log.info("[prefetch/%s] thread started", vm_id)
    while not stop.is_set() and not _shutdown_flag["stop"]:
        try:
            creator_id = claims.claim_next(vm_id)
        except Exception as e:
            log.exception("[prefetch/%s] claim_next failed: %s", vm_id, e)
            time.sleep(30)
            continue

        if creator_id is None:
            status = claims.pool_status()
            if status["pending"] == 0 and status["active"] == 0:
                # Truly empty pool, signal main loop to exit.
                try:
                    q.put(_POOL_EMPTY, timeout=10)
                except queue.Full:
                    pass
                log.info("[prefetch/%s] pool empty, exiting", vm_id)
                return
            # Others are processing; wait and retry.
            log.info(
                "[prefetch/%s] nothing claimable (%d pending, %d active); sleeping %ds",
                vm_id, status["pending"], status["active"], config.CLAIM_POLL_SECONDS,
            )
            time.sleep(config.CLAIM_POLL_SECONDS)
            continue

        _shutdown_flag["prefetch_creator"] = creator_id
        log.info("[prefetch/%s] pulling %s", vm_id, creator_id)
        t0 = time.time()
        try:
            _pull_inputs(creator_id)
        except Exception as e:
            log.exception("[prefetch/%s] pull failed for %s: %s", vm_id, creator_id, e)
            claims.release(creator_id, vm_id)
            _shutdown_flag["prefetch_creator"] = None
            # Best-effort cleanup of any partial pull
            try:
                cleanup_local(creator_id)
            except Exception:
                pass
            continue

        pull_sec = int(time.time() - t0)
        log.info("[prefetch/%s] pulled %s in %ds", vm_id, creator_id, pull_sec)

        # Block until main loop consumes this entry; keeps at most 1 creator
        # pre-pulled ahead.
        while not stop.is_set() and not _shutdown_flag["stop"]:
            try:
                q.put(creator_id, timeout=5)
                # Ownership transferred to main thread; clear our tracking.
                _shutdown_flag["prefetch_creator"] = None
                break
            except queue.Full:
                continue
    log.info("[prefetch/%s] thread stopping", vm_id)


def _wipe_stale_local() -> None:
    """Remove leftover local creator data from a previous pod run.

    Called at orchestrator startup. If the previous pod crashed (SIGKILL,
    OOM, host failure), worker.cleanup_local never ran for the in-flight
    creator, so 3-30 GB of media may sit on the volume. We clear it now
    because we'll re-pull whatever we re-claim anyway.

    We DO NOT remove the model cache (DOVER, laughter-detection, HF cache);
    those are expensive to re-download and live elsewhere.
    """
    import shutil
    for sub in ("segments", "transcripts", "labeled", "metadata", "features_out",
                "clip_embeddings_out"):
        d = config.SHARED_ROOT / sub
        if not d.exists():
            continue
        try:
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("startup wipe failed for %s: %s", d, e)
    log.info("[startup] wiped any leftover local creator data")


def run(vm_id: str, device: str = "cuda", max_creators: int | None = None) -> None:
    """Main pod loop with prefetch."""
    _install_signal_handlers()
    _shutdown_flag["vm_id"] = vm_id

    config.ensure_local_dirs()
    _wipe_stale_local()
    log.info(
        "[%s] Orchestrator starting (device=%s, max_creators=%s, prefetch=on)",
        vm_id, device, max_creators,
    )
    _discord(f"[features/{vm_id}] started on {device}, prefetch enabled")

    prefetch_queue: queue.Queue = queue.Queue(maxsize=1)
    prefetch_stop = threading.Event()
    prefetch_thread = threading.Thread(
        target=_prefetch_loop,
        args=(vm_id, prefetch_queue, prefetch_stop),
        name=f"prefetch-{vm_id}",
        daemon=True,
    )
    prefetch_thread.start()

    done = 0
    while not _shutdown_flag["stop"]:
        if max_creators is not None and done >= max_creators:
            log.info("[%s] Hit max_creators=%d, stopping", vm_id, max_creators)
            break

        # Wait for prefetch to deliver a pre-pulled creator.
        try:
            item = prefetch_queue.get(timeout=30)
        except queue.Empty:
            if not prefetch_thread.is_alive():
                log.error("[%s] prefetch thread died; exiting", vm_id)
                break
            continue

        if item == _POOL_EMPTY:
            log.info("[%s] Pool empty (signal from prefetch). Exiting.", vm_id)
            _discord(f"[features/{vm_id}] pool empty. exiting. processed {done}.")
            break

        creator_id = item
        _shutdown_flag["current_creator"] = creator_id
        try:
            stats = process_creator(
                creator_id, vm_id, device=device, skip_pull=True
            )
            done += 1
            _discord(
                f"[features/{vm_id}] {creator_id} done — "
                f"{stats['segments']} segs in {stats['wall_time_seconds']}s. "
                f"Processed: {done}"
            )
        except Exception as e:
            log.exception("[%s] process_creator(%s) failed: %s", vm_id, creator_id, e)
            try:
                claims.release(creator_id, vm_id)
            except Exception as e2:
                log.error("release after failure also failed: %s", e2)
            # Defensive: wipe partial local data for the failed creator so
            # disk doesn't fill across a flurry of failures.
            try:
                cleanup_local(creator_id)
            except Exception:
                pass
            _discord(f"[features/{vm_id}] {creator_id} FAILED: {e}")
        finally:
            _shutdown_flag["current_creator"] = None
            # Bound GPU memory growth across creators.
            _empty_cuda_cache()

    # Graceful shutdown: stop prefetch, release any orphaned claim it made.
    prefetch_stop.set()
    prefetch_thread.join(timeout=30)

    # Case 1: prefetch had already pulled a creator and put it on the queue,
    # but main exited before consuming it. Release the claim, wipe local data.
    try:
        remaining = prefetch_queue.get_nowait()
        if remaining not in (_POOL_EMPTY, None):
            log.info("[%s] Releasing unused pre-pulled claim %s", vm_id, remaining)
            claims.release(remaining, vm_id)
            try:
                cleanup_local(remaining)
            except Exception:
                pass
    except queue.Empty:
        pass

    # Case 2: prefetch was mid-pull when shutdown started (held claim, never
    # put on queue). Release that claim too; otherwise it orphans until TTL.
    in_flight = _shutdown_flag.get("prefetch_creator")
    if in_flight:
        log.info("[%s] Releasing in-flight (mid-pull) prefetch claim %s", vm_id, in_flight)
        try:
            claims.release(in_flight, vm_id)
        except Exception as e:
            log.warning("release of in-flight prefetch failed: %s", e)
        try:
            cleanup_local(in_flight)
        except Exception:
            pass
        _shutdown_flag["prefetch_creator"] = None

    _discord(f"[features/{vm_id}] exiting. {done} creators processed this session.")
    log.info("[%s] Exit. Processed %d creators this session.", vm_id, done)
