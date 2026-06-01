"""Two-tier per-pod extraction pipeline (inside one creator's processing).

Layout during a creator's processing:

  (pre-pipeline: worker._pull_inputs blocks until R2 pull completes)

  [CPU extractor thread(s)]   text + audio_speech + voice_quality + structural
          |                    + creator_context
          v
      cpu_queue  (bounded)
          |
          v
  [GPU extractor thread]      audio_events + audio_emotion + visual (batched)
          |
      gpu_queue  (bounded)
          |
          v
  [collector]              merges CPU + GPU rows into one dict per segment

  (post-pipeline: novelty.apply runs over collected CLIP embeddings)

Thread safety:
  - Model objects (Wav2Vec, CLIP, DOVER, etc.) are loaded once and only
    touched by the GPU thread.
  - Librosa / parselmouth / numpy operations release the GIL, so multiple
    CPU threads can run in parallel on the 8 vCPUs of the 4090 pod.
  - Inter-thread communication is via queue.Queue (thread-safe).
  - Sentinel value `None` signals end-of-stream to downstream threads.

Batching for GPU:
  - CLIP + DOVER run with a segment batch of CLIP_BATCH_SIZE.
  - The GPU thread collects that many items off cpu_queue before flushing.
  - On SENTINEL, a final smaller flush completes the creator.

Note on prefetch:
  The initial R2 pull is blocking (worker._pull_inputs). A future optimisation
  would add a prefetch thread that downloads the next creator's segments
  while the current one is extracting. Not implemented in v2.1; the pool
  model means the latency cost is ~1-2 min per creator, ~5% of wall time.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config

log = logging.getLogger("clipwhy.features.pipeline")


# ── Dataclass passed between threads ────────────────────────────────────────
@dataclass
class SegmentJob:
    """Everything a segment needs to carry through the pipeline."""
    segment_id: str
    video_id: str
    creator_id: str
    category: str
    segment_index: int
    label: int
    # Metadata for structural features, resolved from labeled CSV:
    start_time: float
    end_time: float
    duration: float
    video_duration: float
    position_ratio: float
    max_segment_index_in_video: int
    # Paths (local, after prefetch):
    audio_path: Path
    video_path: Optional[Path]   # None if MP4 missing (rare)
    transcript_path: Path
    transcript_text: str
    # Populated by CPU thread, consumed by GPU thread:
    cpu_features: dict = None
    # Populated by GPU thread, consumed by collector:
    gpu_features: dict = None
    # CLIP embedding (raw, pre-PCA) held here for the novelty post-hoc pass:
    clip_embedding_raw: Optional["np.ndarray"] = None


SENTINEL = None  # end-of-stream marker


# ── Pipeline runner ─────────────────────────────────────────────────────────
class Pipeline:
    """Run the three-thread pipeline for one creator.

    Usage:
        p = Pipeline(
            cpu_extractor=cpu_fn,
            gpu_extractor=gpu_fn,
            cpu_threads=2,
        )
        rows, clip_embeddings = p.run(jobs)
    """

    def __init__(
        self,
        cpu_extractor: Callable[[SegmentJob], dict],
        gpu_extractor_batch: Callable[[list[SegmentJob]], list[tuple[dict, "np.ndarray"]]],
        cpu_threads: int = config.CPU_WORKER_THREADS,
        batch_size: int = config.CLIP_BATCH_SIZE,
    ):
        self.cpu_extractor = cpu_extractor
        self.gpu_extractor_batch = gpu_extractor_batch
        self.cpu_threads = cpu_threads
        self.batch_size = batch_size

    def run(self, jobs: list[SegmentJob]) -> tuple[list[dict], dict[tuple[str, int], "np.ndarray"]]:
        """Process all jobs. Returns (rows, clip_embeddings_by_segment).

        `rows` is a list of merged {cpu + gpu + keys + label} dicts ready for
        CSV emit. `clip_embeddings_by_segment` maps (video_id, segment_index)
        to the raw CLIP embedding vector for the novelty post-hoc pass.
        """
        cpu_queue: queue.Queue = queue.Queue(maxsize=config.CPU_TO_GPU_QUEUE_MAXSIZE)
        gpu_queue: queue.Queue = queue.Queue(maxsize=config.GPU_OUT_QUEUE_MAXSIZE)

        # Shared error flag: if any thread raises, stop the world.
        error_holder: list[BaseException] = []
        error_lock = threading.Lock()

        def record_error(e: BaseException):
            with error_lock:
                error_holder.append(e)

        # ── CPU stage ───────────────────────────────────────────────────────
        # We split the input jobs across N CPU threads round-robin, each
        # emitting its own completions into the shared cpu_queue.
        job_iter_lock = threading.Lock()
        job_idx = [0]

        def next_job() -> Optional[SegmentJob]:
            with job_iter_lock:
                if job_idx[0] >= len(jobs):
                    return None
                j = jobs[job_idx[0]]
                job_idx[0] += 1
                return j

        def cpu_worker(tid: int):
            try:
                while True:
                    j = next_job()
                    if j is None:
                        return
                    t0 = time.perf_counter()
                    try:
                        j.cpu_features = self.cpu_extractor(j)
                    except Exception as e:
                        log.exception("[cpu#%d] extract failed on %s: %s", tid, j.segment_id, e)
                        j.cpu_features = {}  # emit zeros; GPU still runs on other data
                    dt = time.perf_counter() - t0
                    log.debug("[cpu#%d] %s done in %.3fs", tid, j.segment_id, dt)
                    # Wrap put in try/except so a single failed enqueue doesn't
                    # silently drop work or hang the worker.
                    try:
                        cpu_queue.put(j, timeout=300)
                    except queue.Full:
                        log.error(
                            "[cpu#%d] cpu_queue stuck full for >5min; dropping %s",
                            tid, j.segment_id,
                        )
                    except Exception as e:
                        log.exception("[cpu#%d] cpu_queue.put failed: %s", tid, e)
            except BaseException as e:
                record_error(e)

        cpu_workers = [
            threading.Thread(target=cpu_worker, args=(i,), name=f"cpu-{i}", daemon=True)
            for i in range(self.cpu_threads)
        ]
        for w in cpu_workers:
            w.start()

        # ── CPU finisher: when all CPU threads exit, send sentinel to GPU queue ─
        def cpu_joiner():
            for w in cpu_workers:
                w.join()
            cpu_queue.put(SENTINEL)

        joiner_thread = threading.Thread(target=cpu_joiner, name="cpu-joiner", daemon=True)
        joiner_thread.start()

        # ── GPU stage ───────────────────────────────────────────────────────
        # Track segment progress for periodic logging during long extractions.
        items_processed_holder = [0]
        total_jobs = len(jobs)

        def gpu_worker():
            try:
                batch: list[SegmentJob] = []

                def flush():
                    if not batch:
                        return
                    t0 = time.perf_counter()
                    try:
                        results = self.gpu_extractor_batch(batch)
                    except Exception as e:
                        log.exception("[gpu] batch extract failed (size=%d): %s", len(batch), e)
                        results = [({}, None)] * len(batch)
                    dt = time.perf_counter() - t0
                    log.debug("[gpu] batch %d in %.3fs (%.3fs/seg)", len(batch), dt, dt / len(batch))
                    for j, (feats, clip_raw) in zip(batch, results):
                        j.gpu_features = feats
                        j.clip_embedding_raw = clip_raw
                        gpu_queue.put(j)
                    # Periodic progress: log every ~50 segments so long
                    # extractions show signs of life (was previously silent
                    # between BEGIN and DONE for the whole creator).
                    items_processed_holder[0] += len(batch)
                    n = items_processed_holder[0]
                    prev = n - len(batch)
                    if (n // 50) > (prev // 50):
                        log.info(
                            "[pipeline] %d/%d segments processed (%.1f%%)",
                            n, total_jobs, 100.0 * n / max(total_jobs, 1),
                        )
                    batch.clear()

                while True:
                    try:
                        item = cpu_queue.get(timeout=600)
                    except queue.Empty:
                        # CPU joiner should have pushed SENTINEL; if 10 min
                        # passes with nothing, assume the producer is dead.
                        log.error("[gpu] cpu_queue idle >10min; flushing + exiting")
                        flush()
                        gpu_queue.put(SENTINEL)
                        return
                    if item is SENTINEL:
                        flush()
                        gpu_queue.put(SENTINEL)
                        return
                    batch.append(item)
                    if len(batch) >= self.batch_size:
                        flush()
            except BaseException as e:
                record_error(e)
                # CRITICAL: ensure collector does not deadlock on gpu_queue.get().
                # We always push a sentinel even on unhandled errors so the
                # collector can drain and exit, and the re-raise in run()
                # surfaces the real exception to the caller.
                try:
                    gpu_queue.put_nowait(SENTINEL)
                except queue.Full:
                    # Drain one slot and retry; worst case we drop one result
                    # but the exception we're about to re-raise is the real issue.
                    try:
                        gpu_queue.get_nowait()
                        gpu_queue.put_nowait(SENTINEL)
                    except queue.Empty:
                        pass

        gpu_thread = threading.Thread(target=gpu_worker, name="gpu", daemon=True)
        gpu_thread.start()

        # ── Collector ───────────────────────────────────────────────────────
        rows: list[dict] = []
        clip_embeddings: dict[tuple[str, int], "np.ndarray"] = {}

        while True:
            j = gpu_queue.get()
            if j is SENTINEL:
                break
            rows.append(self._merge_row(j))
            if j.clip_embedding_raw is not None:
                clip_embeddings[(j.video_id, j.segment_index)] = j.clip_embedding_raw

        # Final barrier: ensure GPU thread exited cleanly.
        gpu_thread.join(timeout=30)
        joiner_thread.join(timeout=30)

        if error_holder:
            raise error_holder[0]

        return rows, clip_embeddings

    @staticmethod
    def _merge_row(j: SegmentJob) -> dict:
        row = {
            "segment_id": j.segment_id,
            "video_id": j.video_id,
            "creator_id": j.creator_id,
            "category": j.category,
            "segment_index": j.segment_index,
            "label": j.label,
        }
        if j.cpu_features:
            row.update(j.cpu_features)
        if j.gpu_features:
            row.update(j.gpu_features)
        return row
