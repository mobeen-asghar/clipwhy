"""Per-creator pipeline.

Given a claimed creator_id, this worker:
  1. Pulls labeled CSV, baselines CSV, all segment audio/video/transcripts from R2.
  2. Builds a list of SegmentJob instances.
  3. Starts a CPU extractor thread-pool and a GPU extractor thread.
  4. Runs all segments through the pipeline.
  5. Runs the post-hoc segment_novelty pass over the collected CLIP embeddings.
  6. Writes per-creator features CSV to R2.
  7. Marks the creator done and releases the claim.
  8. Cleans local disk.

Failure policy:
  - If the whole creator fails, we do NOT mark done; the claim will expire via
    TTL and another pod can retry.
  - If individual segments fail within the pipeline, their feature rows get
    zero-filled and tagged with a suffix on features_version.
"""
import concurrent.futures
import csv
import io
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import claims, config, r2_client
from .extractors import audio_emotion, audio_events, audio_speech
from .extractors import creator_context as creator_ctx
from .extractors import novelty, structural, text, visual, voice_quality
from .pipeline import Pipeline, SegmentJob

log = logging.getLogger("clipwhy.features.worker")


# ── Heartbeat thread (keeps claim alive while we work) ──────────────────────
class Heartbeat:
    def __init__(self, creator_id: str, vm_id: str):
        self.creator_id = creator_id
        self.vm_id = vm_id
        self._stop = False
        self._thread = None

    def start(self):
        import threading

        def loop():
            # Renew immediately on start so the claim's heartbeat reflects an
            # active pod from t=0, not t=15 min. Then sleep+renew in a loop.
            try:
                claims.renew(self.creator_id, self.vm_id)
            except Exception as e:
                log.warning("[%s] heartbeat first renew failed: %s", self.vm_id, e)
            while not self._stop:
                # Use Event.wait() so stop() interrupts the sleep instantly.
                if self._stop_event.wait(timeout=config.CLAIM_RENEW_SECONDS):
                    return
                try:
                    claims.renew(self.creator_id, self.vm_id)
                except Exception as e:
                    log.warning("[%s] heartbeat renew failed: %s", self.vm_id, e)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=loop, name="heartbeat", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ── Main entrypoint ─────────────────────────────────────────────────────────
def process_creator(
    creator_id: str,
    vm_id: str,
    device: str = "cuda",
    skip_pull: bool = False,
) -> dict:
    """Full extraction for one creator.

    Args:
        skip_pull: when True, assume _pull_inputs has already been run for
            this creator (e.g., by the prefetch thread in orchestrator).

    Returns a stats dict. Raises on hard failure (orchestrator will not mark done).
    """
    t_start = time.time()
    log.info("[%s] BEGIN %s", vm_id, creator_id)

    hb = Heartbeat(creator_id, vm_id)
    hb.start()

    try:
        # 1. Pull inputs (unless prefetch already did)
        if not skip_pull:
            _pull_inputs(creator_id)
        else:
            log.info("[%s] %s: pull skipped (pre-fetched)", vm_id, creator_id)

        # 2. Build segment jobs
        jobs = _build_jobs(creator_id)
        log.info("[%s] %s: %d segments to process", vm_id, creator_id, len(jobs))
        if not jobs:
            raise RuntimeError(f"No segments for {creator_id}")

        # 3. Lazy-load models once per creator (amortised across ~1000 segments)
        models = _load_models(device=device)

        # 4. Define per-segment CPU extractor closure
        def cpu_extract(job: SegmentJob) -> dict:
            out = {}
            out.update(text.extract(job))
            out.update(audio_speech.extract(job))
            out.update(voice_quality.extract(job))
            out.update(structural.extract(job))
            out.update(creator_ctx.extract(job))
            return out

        # 5. Define batched GPU extractor closure
        def gpu_extract_batch(batch: list[SegmentJob]) -> list[tuple[dict, "np.ndarray"]]:
            # Each sub-extractor returns per-segment dicts; visual also returns
            # raw CLIP embedding for novelty post-hoc.
            event_outs = [audio_events.extract(j, models) for j in batch]
            emotion_outs = [audio_emotion.extract(j, models) for j in batch]
            # visual.extract_batch returns a list of (feature_dict, clip_raw)
            # tuples, one per job. Split them apart here.
            visual_results = visual.extract_batch(batch, models)
            visual_outs = [r[0] for r in visual_results]
            clip_raws = [r[1] for r in visual_results]

            results = []
            for ev, em, vis, clip_raw in zip(event_outs, emotion_outs, visual_outs, clip_raws):
                merged = {}
                merged.update(ev)
                merged.update(em)
                merged.update(vis)
                results.append((merged, clip_raw))
            return results

        # 6. Run pipeline
        p = Pipeline(
            cpu_extractor=cpu_extract,
            gpu_extractor_batch=gpu_extract_batch,
        )
        rows, clip_embeddings = p.run(jobs)

        # 7. Post-hoc novelty pass
        rows = novelty.apply(rows, clip_embeddings)

        # 8. Add metadata columns and write CSV
        features_version = _finalise_rows(rows)
        local_csv = config.LOCAL_FEATURES_OUT / f"{creator_id}_features.csv"
        _write_csv(rows, local_csv)

        # 9a. Save raw CLIP embeddings as npz for post-extraction PCA fit.
        # We do this BEFORE pushing the CSV so a failure here surfaces cleanly:
        # if we fail to save embeddings the creator stays unmarked, the claim
        # expires via TTL, and the next pod retries from scratch.
        _save_and_push_clip_embeddings(creator_id, clip_embeddings)

        # 9b. Push features CSV to R2 (atomic: done marker only after both up)
        r2_client.rclone_copyto_up(
            local_csv,
            f"{config.R2_FEATURES_PREFIX}/{creator_id}_features.csv",
        )

        # 10. Mark done
        stats = {
            "segments": len(rows),
            "features_version": features_version,
            "wall_time_seconds": int(time.time() - t_start),
        }
        claims.mark_done(creator_id, vm_id, stats)
        log.info(
            "[%s] DONE %s: %d segments in %.0fs",
            vm_id, creator_id, len(rows), time.time() - t_start,
        )

        return stats

    finally:
        hb.stop()
        _cleanup_local(creator_id)


# ── Input acquisition ──────────────────────────────────────────────────────
def _pull_inputs(creator_id: str) -> None:
    """Pull labeled CSV + baselines + all segment media + transcripts for this creator."""
    config.ensure_local_dirs()

    # 1a. Labeled CSV
    labeled_remote = f"{config.R2_LABELED_PREFIX}/{creator_id}_segments.csv"
    labeled_local = config.LOCAL_LABELED / f"{creator_id}_segments.csv"
    labeled_local.parent.mkdir(parents=True, exist_ok=True)
    blob = r2_client.get(labeled_remote)
    if blob is None:
        raise RuntimeError(f"Labeled CSV missing on R2 for {creator_id}")
    labeled_local.write_bytes(blob)

    # 1b. Baselines (for creator context — though v2.1 only keeps category)
    baselines_local = config.LOCAL_METADATA / f"{creator_id}_baselines.csv"
    baselines_local.parent.mkdir(parents=True, exist_ok=True)
    blob = r2_client.get(f"{config.R2_METADATA_PREFIX}/{creator_id}_baselines.csv")
    if blob is not None:
        baselines_local.write_bytes(blob)

    # 1c. All segment audio/video/thumbnails + transcripts for every video_id
    df = pd.read_csv(labeled_local)
    video_ids = sorted(df["video_id"].unique())
    log.info("  pulling media for %d videos", len(video_ids))

    for vid in video_ids:
        r2_client.rclone_copy_down(
            f"{config.R2_SEGMENTS_PREFIX}/{vid}/",
            config.LOCAL_SEGMENTS / vid,
        )
        r2_client.rclone_copy_down(
            f"{config.R2_TRANSCRIPTS_PREFIX}/{vid}/",
            config.LOCAL_TRANSCRIPTS / vid,
        )

    # 1d. Pre-transcode any AV1-encoded segment MP4s to H.264 alongside the
    # original. DOVER's decord backend can't decode AV1; visual extractors
    # pick up the h264 sibling when it exists. Parallel transcoding here
    # is hidden by prefetch thread parallelism with extraction.
    _pretranscode_av1_segments(video_ids)


def _pretranscode_av1_segments(video_ids: list[str], max_workers: int = 2) -> None:
    """For each segment MP4 that is not H.264, transcode to a sibling
    <segment>.h264.mp4 using GPU-accelerated NVDEC AV1 decode + NVENC H.264
    encode on the RTX 4090.

    Why GPU:
      - libdav1d (CPU AV1 decode) at parallel=4 with libx264 internal threads
        oversubscribed the 8 vCPUs and ran at ~30s/segment (observed).
      - av1_cuvid + h264_nvenc runs at ~1s/segment per ffmpeg invocation,
        with 2 parallel sessions on a 4090's NVENC engine.
    """
    candidates: list[Path] = []
    missing_video_dirs: list[str] = []
    for vid in video_ids:
        vdir = config.LOCAL_SEGMENTS / vid / "video"
        if not vdir.exists():
            missing_video_dirs.append(vid)
            continue
        n_before = len(candidates)
        for mp4 in sorted(vdir.glob("segment_*.mp4")):
            if mp4.name.endswith(".h264.mp4"):
                continue
            if mp4.with_suffix(".h264.mp4").exists():
                continue
            candidates.append(mp4)
        if len(candidates) == n_before:
            missing_video_dirs.append(vid)

    if missing_video_dirs:
        log.warning(
            "  no video MP4s for %d/%d videos (%s); visual features will be 0 for those segments",
            len(missing_video_dirs), len(video_ids),
            ",".join(missing_video_dirs[:3]) + ("..." if len(missing_video_dirs) > 3 else ""),
        )

    if not candidates:
        log.warning("  no segment MP4s to transcode (all videos missing or empty)")
        return

    t0 = time.time()
    log.info("  transcoding %d segment MP4s -> H.264 via NVDEC+NVENC (parallel=%d)",
             len(candidates), max_workers)

    # Pixel formats decord can read reliably. Anything outside this set
    # gets transcoded even if the codec is already h264.
    _DECORD_OK_PIX_FMTS = {"yuv420p", "yuvj420p", "yuv422p", "yuv444p"}

    def transcode_one(src: Path) -> tuple[Path, bool, str]:
        """Returns (path, was_transcoded, reason).

        Transcodes when:
          - codec != h264, OR
          - pix_fmt is unknown/missing (decord can't read these even if codec is h264)
        """
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,pix_fmt",
                    "-of", "csv=p=0",
                    str(src),
                ],
                check=False, timeout=10,
                capture_output=True, text=True,
            )
            line = probe.stdout.strip()
            # ffprobe csv output: "h264,yuv420p" (codec then pix_fmt)
            parts = [p.strip() for p in line.split(",")]
            codec = parts[0] if len(parts) >= 1 else ""
            pix_fmt = parts[1] if len(parts) >= 2 else ""
            if not codec:
                return src, False, "no_codec"

            # Skip transcode only if it's already H.264 with a decord-friendly pix_fmt.
            if codec == "h264" and pix_fmt in _DECORD_OK_PIX_FMTS:
                return src, False, "already_clean_h264"

            dst = src.with_suffix(".h264.mp4")
            tmp = dst.with_suffix(".partial.mp4")

            # GPU path: NVDEC decode + NVENC encode. Frames pass through
            # CPU memory between stages (no -hwaccel_output_format cuda) so
            # ffmpeg can normalise pix_fmt to yuv420p even when the source
            # advertises pix_fmt=unknown. This ~doubles per-transcode time
            # vs full-GPU pipeline (~1s vs ~0.5s) but works on all sources.
            decoder = "av1_cuvid" if codec == "av1" else f"{codec}_cuvid"
            gpu_cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-hwaccel", "cuda",
                "-c:v", decoder,
                "-i", str(src),
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-rc", "constqp",
                "-qp", "20",
                "-pix_fmt", "yuv420p",   # explicit; guards against unknown source pix_fmt
                "-an",
                str(tmp),
            ]
            r = subprocess.run(
                gpu_cmd, check=False, timeout=120,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(dst)
                return src, True, "gpu"

            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

            # CPU fallback. Same explicit pix_fmt.
            cpu_cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "20",
                "-threads", "2",
                "-pix_fmt", "yuv420p",
                "-an",
                str(tmp),
            ]
            r = subprocess.run(
                cpu_cmd, check=False, timeout=180,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(dst)
                return src, True, "cpu_fallback"
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            return src, False, "ffmpeg_failed"
        except Exception as e:
            return src, False, f"exception:{type(e).__name__}"

    done = 0
    transcoded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(transcode_one, src) for src in candidates]
        for fut in concurrent.futures.as_completed(futures):
            _, did, _ = fut.result()
            done += 1
            if did:
                transcoded += 1
            if done % 500 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                log.info(
                    "  transcode progress: %d/%d (%d transcoded), rate %.2f segs/sec",
                    done, len(candidates), transcoded, rate,
                )
    dt = time.time() - t0
    log.info(
        "  transcoded %d/%d segment MP4s in %.0fs (%.2fs/seg)",
        transcoded, len(candidates), dt,
        dt / max(len(candidates), 1),
    )


# ── Build jobs from labeled CSV ─────────────────────────────────────────────
def _build_jobs(creator_id: str) -> list[SegmentJob]:
    labeled = pd.read_csv(config.LOCAL_LABELED / f"{creator_id}_segments.csv")
    # Compute max segment index per video (for is_last_segment)
    max_idx = labeled.groupby("video_id")["segment_index"].max().to_dict()

    jobs: list[SegmentJob] = []
    for _, row in labeled.iterrows():
        vid = row["video_id"]
        idx = int(row["segment_index"])
        audio_path = config.LOCAL_SEGMENTS / vid / "audio" / f"segment_{idx:03d}.wav"
        video_path = config.LOCAL_SEGMENTS / vid / "video" / f"segment_{idx:03d}.mp4"
        transcript_path = config.LOCAL_TRANSCRIPTS / vid / f"segment_{idx:03d}.json"

        # Load transcript text (fast, small)
        transcript_text = str(row.get("transcript_text", "") or "")

        jobs.append(SegmentJob(
            segment_id=row["segment_id"],
            video_id=vid,
            creator_id=creator_id,
            category=row["category"],
            segment_index=idx,
            label=int(row["label"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            duration=float(row["duration"]),
            video_duration=float(row["video_duration"]),
            position_ratio=float(row["position_ratio"]),
            max_segment_index_in_video=int(max_idx[vid]),
            audio_path=audio_path,
            video_path=video_path if video_path.exists() else None,
            transcript_path=transcript_path,
            transcript_text=transcript_text,
        ))
    return jobs


# ── Model loading (once per pod in the worker module scope via caching) ─────
_MODELS_CACHE = {}

def _load_models(device: str = "cuda") -> dict:
    """Lazy-load all pretrained models once per pod.

    Cached in module-level dict so subsequent calls (next creator) are no-ops.
    """
    if _MODELS_CACHE:
        return _MODELS_CACHE
    log.info("Loading models on device=%s ...", device)
    _MODELS_CACHE.update({
        "yamnet": audio_events.load_yamnet(device=device),
        "gillick": audio_events.load_gillick_laughter(device=device),
        "wav2vec_emotion": audio_emotion.load_wav2vec_emotion(device=device),
        "clip": visual.load_clip(device=device),
        "dover": visual.load_dover(device=device),
        "transnet": visual.load_transnet(device=device),
        "scrfd": visual.load_scrfd(device=device),
    })
    log.info("Models loaded.")
    return _MODELS_CACHE


# ── Finalise + write ────────────────────────────────────────────────────────
def _finalise_rows(rows: list[dict]) -> str:
    """Attach metadata columns, determine features_version suffix if partial."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    partial = False
    for r in rows:
        r.setdefault("extracted_at", now)
        # Fill missing feature columns with 0 (ensures CSV shape is consistent).
        for col in config.FEATURE_COLUMNS:
            if col not in r:
                r[col] = 0.0
                partial = True
    version = config.FEATURES_VERSION + ("-partial" if partial else "")
    for r in rows:
        r["features_version"] = version
    return version


def _save_and_push_clip_embeddings(
    creator_id: str,
    clip_embeddings: dict[tuple[str, int], np.ndarray],
) -> None:
    """Persist per-creator raw CLIP embeddings as a compressed npz and push to R2.

    Why: clip_pca_00..31 columns in the per-creator features CSV are written
    as zeros during extraction because PCA can only be fit after the
    train/val/test split is decided (post-extraction). Saving the raw 768-d
    embeddings inline now lets the post-extraction PCA fit step work directly
    from R2 without re-pulling MP4s and re-running CLIP (~3-4h saved).

    File contents:
      - segment_keys: shape (N,) array of "<video_id>|<segment_index>" strings
      - embeddings:   shape (N, 768) float32 array, mean-pooled across the 5
                      sampled frames per segment
    """
    if not clip_embeddings:
        log.warning(
            "[clip_emb] no CLIP embeddings collected for %s; skipping npz",
            creator_id,
        )
        return

    items = sorted(clip_embeddings.items())  # by (video_id, segment_index)
    keys_arr = np.array([f"{vid}|{idx}" for (vid, idx), _ in items])
    embs_arr = np.stack(
        [np.asarray(emb, dtype=np.float32) for _, emb in items], axis=0
    )

    local_npz = config.LOCAL_CLIP_EMBEDDINGS_OUT / f"{creator_id}_clip_embeddings.npz"
    local_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local_npz, segment_keys=keys_arr, embeddings=embs_arr)

    r2_client.rclone_copyto_up(
        local_npz,
        f"{config.R2_CLIP_EMBEDDINGS_PREFIX}/{creator_id}_clip_embeddings.npz",
    )
    log.info(
        "[clip_emb] pushed %s: %d embeddings (%.1f KB on disk)",
        creator_id, embs_arr.shape[0], local_npz.stat().st_size / 1024.0,
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Reindex to locked column order; any extras are an extractor bug.
    missing = [c for c in config.OUTPUT_COLUMN_ORDER if c not in df.columns]
    extras = [c for c in df.columns if c not in config.OUTPUT_COLUMN_ORDER]
    if missing:
        log.error("Missing columns at write time: %s", missing)
        for c in missing:
            df[c] = 0.0
    if extras:
        log.warning("Extra columns dropped at write time: %s", extras)
    df = df[config.OUTPUT_COLUMN_ORDER]
    df.to_csv(path, index=False)


# ── Cleanup ─────────────────────────────────────────────────────────────────
def cleanup_local(creator_id: str) -> None:
    """Remove all local media + transcripts + features CSV for this creator.

    Safe to call repeatedly (idempotent). Called after features are safely on
    R2 so we bound peak disk to ~2 creators worth of media (current + prefetch).
    Cleans up .h264.mp4 transcoded siblings implicitly because they live
    inside segments/<video_id>/video/ which we rmtree.
    """
    # Remove segments + transcripts for every video in this creator's labeled CSV.
    labeled_local = config.LOCAL_LABELED / f"{creator_id}_segments.csv"
    freed = 0
    if labeled_local.exists():
        try:
            df = pd.read_csv(labeled_local)
            for vid in df["video_id"].unique():
                for d in (config.LOCAL_SEGMENTS / vid, config.LOCAL_TRANSCRIPTS / vid):
                    if d.exists():
                        shutil.rmtree(d, ignore_errors=True)
                        freed += 1
        except Exception as e:
            log.warning("cleanup parse failed for %s: %s", creator_id, e)
        # Also remove the labeled CSV itself (we pull fresh on resume)
        try:
            labeled_local.unlink()
        except Exception:
            pass

    # Features CSV (local). Once R2 has it, the local copy is dead weight.
    features_csv = config.LOCAL_FEATURES_OUT / f"{creator_id}_features.csv"
    if features_csv.exists():
        try:
            features_csv.unlink()
        except Exception:
            pass

    # Per-creator CLIP embeddings npz (also pushed to R2 in process_creator).
    clip_npz = config.LOCAL_CLIP_EMBEDDINGS_OUT / f"{creator_id}_clip_embeddings.npz"
    if clip_npz.exists():
        try:
            clip_npz.unlink()
        except Exception:
            pass

    # Per-creator baselines CSV copy
    baselines_csv = config.LOCAL_METADATA / f"{creator_id}_baselines.csv"
    if baselines_csv.exists():
        try:
            baselines_csv.unlink()
        except Exception:
            pass
    log.info("[cleanup] removed %d local dirs for %s", freed, creator_id)


# Backwards-compat alias (callers use _cleanup_local)
_cleanup_local = cleanup_local
