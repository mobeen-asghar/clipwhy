"""
Per-VM orchestrator: assigns creators to threaded workers, manages progress.

Runs on each VM. Loads this VM's creator assignment, launches N worker
threads, and processes creators until all are done.

Workers share:
  - One Whisper model (loaded once, GPU-locked)
  - One YouTubeAPI instance (thread-safe with lock)
  - One Notifier instance
"""

import logging
import os
import shutil
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config
from . import progress
from .gpu_lock import GPULock
from .worker import CreatorWorker
from .transcriber import load_whisper_model

log = logging.getLogger("clipwhy.orchestrator")


class Orchestrator:
    """Manages worker threads on a single VM."""

    def __init__(self, vm_id: str, api, notify, num_workers: int = None,
                 test_mode: bool = False, phase: str = "all"):
        self.vm_id = vm_id
        self.api = api
        self.notify = notify
        self.num_workers = num_workers or config.WORKERS_PER_VM
        self.test_mode = test_mode
        self.phase = phase  # "all", "cpu", or "gpu"
        self.gpu_lock = GPULock()
        self.whisper_model = None
        self._start_time = time.time()
        self._status_lock = threading.Lock()
        self._last_status_time = time.time()
        self._r2_sync_lock = threading.Lock()
        # Enable R2 sync if R2_ACCESS_KEY is set (RunPod mode)
        self._r2_sync_enabled = bool(os.environ.get("R2_ACCESS_KEY", ""))

    def run(self):
        """Main entry point. Load assignment, launch workers, run to completion."""
        config.ensure_directories()

        # Load creator assignment for this VM
        my_creator_ids = progress.load_assignment(self.vm_id)
        if self.test_mode:
            my_creator_ids = my_creator_ids[:4]  # 4 creators = 1 per worker

        # Load creator data
        creators_csv = config.INPUT_DIR / "creators.csv"
        if not creators_csv.exists():
            raise FileNotFoundError(f"creators.csv not found at {creators_csv}")
        creators_df = pd.read_csv(creators_csv)
        creator_lookup = {
            row["creator_id"]: row.to_dict()
            for _, row in creators_df.iterrows()
        }

        # Filter to creators that need processing
        remaining = []
        for cid in my_creator_ids:
            if cid not in creator_lookup:
                continue
            if not progress.is_done(cid):
                remaining.append(cid)
            elif self.phase == "gpu":
                # GPU phase: include cpu_done creators (they need Whisper)
                meta = progress.load_done_metadata(cid)
                if meta.get("status") == "cpu_done":
                    remaining.append(cid)

        total = len(my_creator_ids)
        done_already = total - len(remaining)

        self.notify.send(
            f"[{self.vm_id}] Starting pipeline (phase={self.phase})\n"
            f"  Assigned: {total} creators\n"
            f"  Already done: {done_already}\n"
            f"  Remaining: {len(remaining)}\n"
            f"  Workers: {self.num_workers}\n"
            f"  Test mode: {self.test_mode}"
        )

        if not remaining:
            self.notify.done(f"[{self.vm_id}] All creators already done")
            return

        # Load Whisper model only if GPU phase is needed
        if self.phase in ("all", "gpu"):
            self.notify.send(f"[{self.vm_id}] Loading Whisper model...")
            self.whisper_model = load_whisper_model()
            self.notify.send(f"[{self.vm_id}] Whisper model ready")
        else:
            self.notify.send(f"[{self.vm_id}] CPU phase, skipping Whisper")

        # Process creators with thread pool
        creator_queue = list(remaining)
        completed_count = done_already
        error_count = 0
        skipped_count = 0

        # Background prefetch pool: pre-pull next creator's GPU inputs from R2
        # while the current creator is in Whisper. Eliminates the per-creator
        # 1-3 minute R2-pull idle gap. Single-worker pool: prefetch one ahead.
        # Disk safety enforced inside _prefetch_creator (skip if free < 12GB).
        prefetch_pool = ThreadPoolExecutor(max_workers=1) if self.phase in ("all", "gpu") else None
        prefetched = set()  # creator_ids whose audio has been prefetched

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}

            # Submit initial batch
            for i in range(min(self.num_workers, len(creator_queue))):
                cid = creator_queue.pop(0)
                creator = creator_lookup[cid]
                worker = CreatorWorker(
                    i, self.vm_id, self.api, self.notify,
                    self.gpu_lock, self.whisper_model, phase=self.phase
                )
                future = pool.submit(worker.process_creator, creator)
                futures[future] = (i, cid)

            # Kick off initial prefetch for what's coming after the in-flight ones.
            if prefetch_pool and creator_queue:
                peek_cid = creator_queue[0]
                if peek_cid not in prefetched:
                    prefetched.add(peek_cid)
                    prefetch_pool.submit(self._prefetch_creator, peek_cid)

            # Process as they complete, assign next
            while futures:
                # Check for completed futures
                done_futures = [f for f in futures if f.done()]

                for future in done_futures:
                    worker_id, cid = futures.pop(future)

                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"status": "error", "error": str(e)}
                        log.exception("Worker %d crashed on %s", worker_id, cid)

                    status = result.get("status", "error")
                    if status in ("done", "cpu_done"):
                        completed_count += 1
                    elif status == "skipped":
                        completed_count += 1
                        skipped_count += 1
                    elif status == "error":
                        error_count += 1

                    # Sync this creator's data to R2 and clean local
                    if self._r2_sync_enabled:
                        self._sync_and_clean_creator(cid, result)

                    # Assign next creator to this worker
                    if creator_queue:
                        next_cid = creator_queue.pop(0)
                        creator = creator_lookup[next_cid]
                        worker = CreatorWorker(
                            worker_id, self.vm_id, self.api, self.notify,
                            self.gpu_lock, self.whisper_model, phase=self.phase
                        )
                        new_future = pool.submit(worker.process_creator, creator)
                        futures[new_future] = (worker_id, next_cid)

                        # Start prefetch for the creator AFTER next, so it's
                        # ready when this worker finishes the one we just submitted.
                        if prefetch_pool and creator_queue:
                            peek_cid = creator_queue[0]
                            if peek_cid not in prefetched:
                                prefetched.add(peek_cid)
                                prefetch_pool.submit(self._prefetch_creator, peek_cid)

                # Periodic status update
                self._maybe_send_status(
                    my_creator_ids, completed_count, error_count,
                    skipped_count, total
                )

                if futures:
                    time.sleep(2)

        if prefetch_pool:
            prefetch_pool.shutdown(wait=False)

        # Final summary
        elapsed = time.time() - self._start_time
        summary = progress.get_progress_summary(my_creator_ids)

        self.notify.done(
            f"[{self.vm_id}] COMPLETE ({elapsed / 3600:.1f}h) | "
            f"{summary['done']} done | {summary['skipped']} skipped | "
            f"{summary['errors']} errors | "
            f"{summary['total_segments']:,} segments | "
            f"{summary['total_pairs']} pairs"
        )

    def _prefetch_creator(self, creator_id: str):
        """Pre-pull next creator's GPU inputs (segment audio + short audio) from R2
        in background so the worker can start Whisper immediately when it gets the
        creator (no R2-pull idle gap).

        Disk safety: skips if free disk on SHARED_ROOT < 12 GB. With 40GB pods,
        this lets us safely keep current creator's audio + next creator's prefetched
        audio simultaneously (~16-20 GB peak) with headroom for transcript writes
        and unexpected size growth.

        Idempotent: worker._pull_gpu_inputs_from_r2 also checks file existence,
        so a partial prefetch is fine - the worker fills in what's missing.
        """
        try:
            free_bytes = shutil.disk_usage(str(config.SHARED_ROOT)).free
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < 12:
                log.info("[prefetch] %s skipped: only %.1f GB free", creator_id, free_gb)
                return

            # Need step4_download cache to know what to pull
            step4 = progress.load_step_cache(creator_id, "step4_download")
            if not step4:
                log.warning("[prefetch] %s: no step4_download cache, can't prefetch", creator_id)
                return

            video_segments = step4.get("video_segments", {})
            matched_short_ids = set(step4.get("matched_short_ids", []))

            # Reuse worker's idempotent pull helper
            worker = CreatorWorker(
                -1, self.vm_id, self.api, self.notify,
                self.gpu_lock, self.whisper_model, phase=self.phase,
            )
            worker._pull_gpu_inputs_from_r2(
                video_segments, matched_short_ids,
                f"[prefetch-{creator_id}]",
            )
            log.info("[prefetch] %s ready: %d videos + %d shorts pre-pulled",
                     creator_id, len(video_segments), len(matched_short_ids))
        except Exception as e:
            log.warning("[prefetch] %s failed: %s", creator_id, e)

    def _sync_and_clean_creator(self, creator_id: str, result: dict):
        """Sync a completed creator's data to R2, then delete local files.
        Thread-safe: only one sync runs at a time to avoid rclone conflicts.

        Phase-aware:
          CPU phase:  segments, raw/long, raw/shorts_*, raw/captions, metadata
                      (CPU produces these; per-video streaming already uploaded
                      most, but this is belt+suspenders)
          GPU phase:  transcripts, pairs, labeled only (segments, raw, metadata
                      are already on R2 from CPU phase, don't re-upload)
          progress/: always
        """
        video_segments = result.get("video_segments", {})
        short_ids = result.get("matched_short_ids", [])
        is_gpu = self.phase == "gpu"

        with self._r2_sync_lock:
            shared = config.SHARED_ROOT
            bucket = "r2:clipwhy-data"

            dirs_to_sync = []

            if not is_gpu:
                # ── CPU-phase artifacts (skip in GPU phase: already on R2) ──
                for vid in video_segments:
                    seg_dir = shared / "segments" / vid
                    if seg_dir.exists():
                        dirs_to_sync.append(
                            (str(seg_dir), f"{bucket}/segments/{vid}")
                        )

                for vid in video_segments:
                    for ext in ("wav", "mp4"):
                        raw_file = shared / "raw" / "long" / f"{vid}.{ext}"
                        if raw_file.exists():
                            dirs_to_sync.append(
                                (str(raw_file.parent), f"{bucket}/raw/long",
                                 f"--include={vid}.*")
                            )
                            break

                captions_dir = shared / "raw" / "captions"
                if captions_dir.exists():
                    dirs_to_sync.append(
                        (str(captions_dir), f"{bucket}/raw/captions")
                    )

                # Metadata is a CPU-phase output
                for subdir in ("metadata",):
                    local = shared / subdir
                    if local.exists():
                        dirs_to_sync.append(
                            (str(local), f"{bucket}/{subdir}")
                        )

            # ── Always sync (small files, written or updated by either phase) ──
            for subdir in ("progress", "pairs", "labeled"):
                local = shared / subdir
                if local.exists():
                    dirs_to_sync.append(
                        (str(local), f"{bucket}/{subdir}")
                    )

            # Transcripts are a GPU-phase output; also safe to always check
            for vid in video_segments:
                tdir = shared / "transcripts" / "segments" / vid
                if tdir.exists():
                    dirs_to_sync.append(
                        (str(tdir), f"{bucket}/transcripts/segments/{vid}")
                    )

            shorts_t_dir = shared / "transcripts" / "shorts"
            if shorts_t_dir.exists() and short_ids:
                for sid in short_ids:
                    f = shorts_t_dir / f"{sid}.json"
                    if f.exists():
                        dirs_to_sync.append(
                            (str(shorts_t_dir),
                             f"{bucket}/transcripts/shorts",
                             f"--include={sid}.json")
                        )

            for entry in dirs_to_sync:
                src, dst = entry[0], entry[1]
                extra = entry[2] if len(entry) > 2 else None
                # --update: skip any file where destination mod time is newer.
                # Prevents stale local state (e.g. pod's progress/ dir cloned
                # hours ago) from downgrading newer R2 markers written by
                # another pod. Root cause of the CR0156/CR0251 rollback bug.
                cmd = [
                    "rclone", "copy", src, dst,
                    "--disable-http2", "--transfers", "8",
                    "--retries", "3", "--retries-sleep", "5s",
                    "--update",
                ]
                if extra:
                    cmd.append(extra)
                try:
                    subprocess.run(cmd, capture_output=True, timeout=300)
                except Exception as e:
                    log.warning("R2 sync failed for %s: %s", src, e)

            # Sync shorts (audio + video) - CPU phase only
            if not is_gpu:
                for sid in short_ids:
                    for subdir in ("shorts_audio", "shorts_video"):
                        for ext in ("wav", "mp4"):
                            f = shared / "raw" / subdir / f"{sid}.{ext}"
                            if f.exists():
                                subprocess.run(
                                    ["rclone", "copy", str(f.parent),
                                     f"{bucket}/raw/{subdir}",
                                     "--disable-http2",
                                     f"--include={sid}.*",
                                     "--retries", "3",
                                     "--update"],
                                    capture_output=True, timeout=120
                                )
                                break

            # Now delete local heavy data for this creator
            for vid in video_segments:
                seg_dir = shared / "segments" / vid
                if seg_dir.exists():
                    shutil.rmtree(seg_dir, ignore_errors=True)
                # Transcripts are small but accumulate, clean per video
                t_dir = shared / "transcripts" / "segments" / vid
                if t_dir.exists():
                    shutil.rmtree(t_dir, ignore_errors=True)
                for ext in ("wav", "mp4"):
                    raw_file = shared / "raw" / "long" / f"{vid}.{ext}"
                    if raw_file.exists():
                        raw_file.unlink(missing_ok=True)

            for sid in short_ids:
                for subdir in ("shorts_audio", "shorts_video"):
                    for ext in ("wav", "mp4"):
                        f = shared / "raw" / subdir / f"{sid}.{ext}"
                        if f.exists():
                            f.unlink(missing_ok=True)
                # Short transcripts also clean per creator
                t_short = shared / "transcripts" / "shorts" / f"{sid}.json"
                if t_short.exists():
                    t_short.unlink(missing_ok=True)

            log.info("R2 sync + clean done for %s", creator_id)

    def _maybe_send_status(self, all_creator_ids, completed, errors,
                           skipped, total):
        """Send periodic status update via Discord."""
        now = time.time()
        with self._status_lock:
            if now - self._last_status_time < config.STATUS_INTERVAL_SEC:
                return
            self._last_status_time = now

        elapsed = now - self._start_time
        elapsed_h = elapsed / 3600
        remaining = total - completed - errors
        rate = completed / elapsed_h if elapsed_h > 0 else 0
        eta_h = remaining / rate if rate > 0 else 0

        summary = progress.get_progress_summary(all_creator_ids)

        if self.phase == "gpu":
            self.notify.send(
                f"[{self.vm_id}] STATUS ({elapsed_h:.1f}h) | "
                f"{completed}/{total} done | "
                f"{summary['total_positives']} positive | "
                f"caption:{summary['total_caption_pairs']} whisper:{summary['total_pairs']} "
                f"dropped:{summary['total_dropped']} | "
                f"ETA {eta_h:.1f}h"
            )
        else:
            self.notify.send(
                f"[{self.vm_id}] STATUS ({elapsed_h:.1f}h) | "
                f"{completed}/{total} done | "
                f"{summary['total_segments']:,} segs | "
                f"{summary['total_pairs']} pairs | "
                f"ETA {eta_h:.1f}h"
            )
