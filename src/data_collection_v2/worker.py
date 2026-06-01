"""
CreatorWorker: processes one creator end-to-end through 7 steps.

Step 1: LIST       - List shorts + longs from channel
Step 2: CAPTIONS   - Download captions, fuzzy match to find pairs
Step 3: METADATA   - Fetch engagement data for ALL shorts NOW (before deletions)
Step 4: DOWNLOAD   - Download matched longs + shorts, segment longs
Step 5: WHISPER    - Transcribe segments + shorts, exact matching (GPU)
Step 6: LABELS     - Apply virality thresholds + label segments from Whisper matches
Step 7: DONE       - Write completion marker

Metadata is collected early (step 3) because YouTube videos get deleted over
time. V1 lost ~414 shorts between collection and Whisper matching. Getting
engagement data immediately after confirming pairs exist prevents data loss.

Each step sends Discord notifications with detailed stats.
"""

import logging
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

from . import config
from . import progress
from .downloader import (
    download_long_audio, download_long_video,
    download_short_audio, download_short_video,
)
from .matcher import find_caption_pairs, get_matched_long_ids, get_matched_short_ids
from .matcher import whisper_match_short_to_segments
from .segmenter import segment_long_video
from .transcriber import transcribe_segments, transcribe_short, load_segment_transcripts
from .engagement import fetch_shorts_metadata, compute_baselines, label_matched_shorts
from .labeler import label_creator_segments

log = logging.getLogger("clipwhy.worker")


class CreatorWorker:
    """Processes one creator at a time through the full pipeline.

    Supports two run phases:
      phase="all"  - Run all steps 1-7 (default, needs GPU)
      phase="cpu"  - Run steps 1-4 only (no GPU needed, download+segment)
      phase="gpu"  - Run steps 5-7 only (needs GPU, picks up from step 4)
    """

    def __init__(self, worker_id, vm_id, api, notify, gpu_lock=None,
                 whisper_model=None, phase="all"):
        self.worker_id = worker_id
        self.vm_id = vm_id
        self.api = api
        self.notify = notify
        self.gpu_lock = gpu_lock
        self.whisper_model = whisper_model
        self.phase = phase

    def _tag(self, creator_id=""):
        """Log/notification prefix."""
        tag = f"[{self.vm_id}/W{self.worker_id}]"
        if creator_id:
            tag += f" {creator_id}"
        return tag

    def process_creator(self, creator: dict) -> dict:
        """Process one creator. Supports phase="all", "cpu", or "gpu".

        phase="cpu":  Steps 1-4 (list, captions, metadata, download). No GPU.
        phase="gpu":  Steps 5-7 (Whisper, labels, done). Needs GPU. Picks up
                      from step 4 completion cache on shared storage.
        phase="all":  Steps 1-7. Full pipeline.

        All state lives on shared storage (GCS). VMs can be deleted and
        recreated; a new VM reads existing files and picks up where the
        old one stopped.
        """
        creator_id = creator["creator_id"]
        channel_id = creator["channel_id"]
        channel_title = creator.get("channel_title", "")
        category = creator.get("category", "")
        comments_disabled = creator.get("comments_disabled", False)

        # Skip fully completed creators (but not cpu_done for GPU phase)
        if progress.is_done(creator_id):
            if self.phase == "gpu":
                # Check if this is cpu_done (needs GPU work) vs fully done
                done_meta = progress.load_done_metadata(creator_id)
                if done_meta.get("status") == "cpu_done":
                    pass  # proceed with GPU phase
                else:
                    return {"status": "already_done"}
            else:
                return {"status": "already_done"}

        # GPU phase: skip creators that haven't finished CPU phase
        if self.phase == "gpu":
            if not progress.load_step_cache(creator_id, "step4_download"):
                return {"status": "skipped", "reason": "cpu_phase_not_done"}

        tag = self._tag(creator_id)
        start_time = time.time()

        try:
            # ── CPU PHASE (steps 1-4) ──────────────────────────────────
            if self.phase in ("all", "cpu"):

                # Step 1: LIST
                short_ids, long_ids = self._step1_list(
                    creator_id, channel_id, tag
                )

                # Step 2: CAPTIONS + FUZZY MATCH
                caption_pairs = self._step2_caption_match(
                    creator_id, short_ids, long_ids, tag
                )
                if not caption_pairs:
                    progress.mark_skipped(creator_id, "no_caption_matches")
                    self.notify.send(f"{tag} SKIPPED (0 caption matches)")
                    return {"status": "skipped", "reason": "no_caption_matches"}

                matched_long_ids = get_matched_long_ids(caption_pairs)
                matched_short_ids = get_matched_short_ids(caption_pairs)

                # Step 3: METADATA (early, before videos get deleted)
                shorts_df, baselines = self._step3_metadata(
                    creator_id, short_ids, comments_disabled, tag
                )

                # Step 4: DOWNLOAD + SEGMENT
                video_segments, _ = self._step4_download(
                    creator_id, matched_long_ids, matched_short_ids, tag
                )

                # Save step 4 completion with data needed by GPU phase
                progress.save_step_cache(creator_id, "step4_download", {
                    "video_segments": video_segments,
                    "matched_short_ids": list(matched_short_ids),
                })

                if self.phase == "cpu":
                    elapsed = time.time() - start_time
                    total_segs = sum(v.get("num_segments", 0) for v in video_segments.values())

                    done_meta = {
                        "status": "cpu_done",
                        "channel_title": channel_title,
                        "category": category,
                        "total_segments": total_segs,
                        "matched_longs": len(video_segments),
                        "pairs_found": len(caption_pairs),
                        "elapsed_min": round(elapsed / 60, 1),
                        "worker_id": self.worker_id,
                        "vm_id": self.vm_id,
                    }
                    progress.mark_done(creator_id, done_meta)

                    self.notify.send(
                        f'{tag} "{channel_title}" CPU DONE ({elapsed / 60:.0f}min) | '
                        f"{len(caption_pairs)} pairs | "
                        f"{len(video_segments)} longs | "
                        f"{total_segs} segments"
                    )
                    return {
                        "status": "cpu_done", **done_meta,
                        "video_segments": video_segments,
                        "matched_short_ids": list(matched_short_ids),
                    }

            # ── GPU PHASE (steps 5-7) ──────────────────────────────────
            if self.phase in ("all", "gpu"):

                # Load state from CPU phase caches on shared storage
                if self.phase == "gpu":
                    step1 = progress.load_step_cache(creator_id, "step1_list")
                    step2 = progress.load_step_cache(creator_id, "step2_captions")
                    step3 = progress.load_step_cache(creator_id, "step3_metadata")
                    step4 = progress.load_step_cache(creator_id, "step4_download")

                    short_ids = step1["short_ids"]
                    caption_pairs = step2
                    matched_short_ids = set(step4["matched_short_ids"])
                    video_segments = step4["video_segments"]
                    baselines = step3.get("baselines", {})

                    # Reload metadata CSV from shared storage
                    shorts_path = config.METADATA_DIR / f"{creator_id}_shorts.csv"
                    shorts_df = pd.read_csv(shorts_path)

                    # Pull segment audio + short audio from R2 to local disk
                    # (they were cleaned off local after CPU phase)
                    self._pull_gpu_inputs_from_r2(
                        video_segments, matched_short_ids, tag
                    )

                # Step 5: WHISPER + EXACT MATCH (GPU)
                whisper_pairs, _ = self._step5_whisper(
                    creator_id, video_segments, matched_short_ids,
                    caption_pairs, tag
                )

                # Free disk immediately: audio is no longer needed after Whisper.
                # Step 6 reads transcripts (already written), not audio.
                self._cleanup_audio_after_whisper(
                    video_segments, matched_short_ids, tag
                )

                # Step 6: LABELS
                label_stats = self._step6_labels(
                    creator_id, category, whisper_pairs,
                    shorts_df, baselines, video_segments, tag
                )

                # Step 7: MARK DONE
                elapsed = time.time() - start_time
                total_segs = label_stats.get("total_segments", 0)
                pos_segs = label_stats.get("positive_segments", 0)
                pos_pct = f"{pos_segs / total_segs * 100:.1f}%" if total_segs > 0 else "N/A"
                caption_count = len(caption_pairs) if caption_pairs else 0
                dropped = caption_count - len(whisper_pairs)
                done_meta = {
                    "channel_title": channel_title,
                    "category": category,
                    "caption_pairs": caption_count,
                    "pairs_found": len(whisper_pairs),
                    "dropped_pairs": dropped,
                    "total_segments": total_segs,
                    "positive_segments": pos_segs,
                    "elapsed_min": round(elapsed / 60, 1),
                    "worker_id": self.worker_id,
                    "vm_id": self.vm_id,
                }
                progress.mark_done(creator_id, done_meta)

                self.notify.send(
                    f'{tag} "{channel_title}" DONE ({elapsed / 60:.0f}min) | '
                    f"caption:{caption_count} whisper:{len(whisper_pairs)} dropped:{dropped} | "
                    f"{total_segs} segs | "
                    f"{pos_segs} positive ({pos_pct}) | "
                    f"{label_stats.get('viral_shorts', 0)} viral"
                )

                return {
                    "status": "done", **done_meta,
                    "video_segments": video_segments,
                    "matched_short_ids": list(matched_short_ids),
                }

        except Exception as e:
            elapsed = time.time() - start_time
            progress.mark_error(creator_id, str(e))
            self.notify.error(
                f"{tag} FAILED after {elapsed / 60:.0f} min: {e}"
            )
            log.exception("Creator %s failed", creator_id)
            return {"status": "error", "error": str(e)}

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _pull_gpu_inputs_from_r2(self, video_segments, matched_short_ids, tag):
        """Pull segment audio + short audio from R2 so GPU phase has files to read.

        CPU phase writes to /workspace/shared/segments/{vid}/audio/ and
        raw/shorts_audio/, then orchestrator._sync_and_clean_creator pushes to
        R2 and deletes local. When the GPU phase starts later on a different
        pod, local disk is empty; this helper pulls only what's needed.
        """
        import os
        import subprocess

        if not os.environ.get("R2_ACCESS_KEY"):
            return  # Not on RunPod, assume files are already local

        bucket = "r2:clipwhy-data"
        n_videos = len(video_segments)
        n_shorts = len(matched_short_ids)
        log.info("%s Pulling inputs from R2: %d videos + %d shorts",
                 tag, n_videos, n_shorts)

        # Pull per-video segment audio. Only audio, not video/thumbnails.
        for vid in video_segments:
            local_dir = config.SEGMENTS_DIR / vid / "audio"
            local_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(local_dir.glob("*.wav")))
            if existing >= video_segments[vid].get("num_segments", 0):
                continue  # already present
            try:
                subprocess.run(
                    ["rclone", "copy",
                     f"{bucket}/segments/{vid}/audio",
                     str(local_dir),
                     "--disable-http2", "--transfers", "16",
                     "--retries", "3", "--retries-sleep", "5s"],
                    capture_output=True, timeout=600, check=False,
                )
            except Exception as e:
                log.warning("R2 pull failed for %s audio: %s", vid, e)

        # Pull matched shorts audio (one file per short, simple loop)
        config.RAW_SHORTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        pulled_shorts = 0
        for sid in matched_short_ids:
            local = config.RAW_SHORTS_AUDIO_DIR / f"{sid}.wav"
            if local.exists() and local.stat().st_size > 100:
                pulled_shorts += 1
                continue
            try:
                subprocess.run(
                    ["rclone", "copyto",
                     f"{bucket}/raw/shorts_audio/{sid}.wav",
                     str(local),
                     "--disable-http2", "--retries", "3"],
                    capture_output=True, timeout=120, check=False,
                )
                if local.exists() and local.stat().st_size > 100:
                    pulled_shorts += 1
            except Exception as e:
                log.warning("R2 pull failed for short %s: %s", sid, e)

        log.info("%s R2 pull done: %d/%d videos audio, %d/%d shorts",
                 tag, n_videos, n_videos, pulled_shorts, n_shorts)

    def _stream_video_to_r2(self, vid, tag):
        """CPU phase: after a long video is downloaded + segmented, immediately
        sync its segments + raw files to R2 and delete local copies.

        Bounds local disk to one video per worker in flight rather than piling
        all of a creator's videos on disk until completion. Idempotent: rclone
        re-upload is a no-op if already there, and local delete is guarded by
        file existence checks.
        """
        import os
        import shutil
        import subprocess

        if not os.environ.get("R2_ACCESS_KEY"):
            return  # not on RunPod mode, keep local-only behaviour

        bucket = "r2:clipwhy-data"
        seg_dir = config.SEGMENTS_DIR / vid
        raw_long = config.RAW_LONG_DIR

        # Sync segments (audio + video + thumbnails) to R2.
        # --update: don't overwrite newer remote files (idempotent safety).
        if seg_dir.exists():
            try:
                subprocess.run(
                    ["rclone", "copy", str(seg_dir),
                     f"{bucket}/segments/{vid}",
                     "--disable-http2", "--transfers", "8",
                     "--retries", "3", "--retries-sleep", "5s",
                     "--update"],
                    capture_output=True, timeout=600, check=False,
                )
            except Exception as e:
                log.warning("%s R2 sync segments %s failed: %s", tag, vid, e)

        # Sync raw long files (audio + video) to R2
        for ext in ("wav", "mp4"):
            f = raw_long / f"{vid}.{ext}"
            if f.exists():
                try:
                    subprocess.run(
                        ["rclone", "copyto", str(f),
                         f"{bucket}/raw/long/{vid}.{ext}",
                         "--disable-http2", "--retries", "3",
                         "--update"],
                        capture_output=True, timeout=900, check=False,
                    )
                except Exception as e:
                    log.warning("%s R2 sync raw/long %s.%s failed: %s", tag, vid, ext, e)

        # Delete local (both raw and segments) to free disk
        for ext in ("wav", "mp4", "m4a", "webm"):
            f = raw_long / f"{vid}.{ext}"
            if f.exists():
                f.unlink(missing_ok=True)
        if seg_dir.exists():
            shutil.rmtree(seg_dir, ignore_errors=True)

        log.info("%s  Streamed %s to R2 + cleaned local", tag, vid)

    def _cleanup_audio_after_whisper(self, video_segments, matched_short_ids, tag):
        """After Whisper finishes, the local segment-audio and short-audio files
        are no longer needed by this creator (Step 6 reads transcripts, not audio).
        Deleting them immediately frees ~1MB x N segments + per-short audio, which
        is typically 5-8 GB per creator on disk. This enables safer prefetching
        or simply keeps the pod under 50% disk during GPU phase.
        """
        import shutil

        freed_audio_dirs = 0
        for vid in video_segments:
            audio_dir = config.SEGMENTS_DIR / vid / "audio"
            if audio_dir.exists():
                shutil.rmtree(audio_dir, ignore_errors=True)
                freed_audio_dirs += 1

        freed_shorts = 0
        for sid in matched_short_ids:
            f = config.RAW_SHORTS_AUDIO_DIR / f"{sid}.wav"
            if f.exists():
                f.unlink(missing_ok=True)
                freed_shorts += 1

        log.info("%s Post-Whisper cleanup: %d audio dirs + %d short files freed",
                 tag, freed_audio_dirs, freed_shorts)

    def _stream_short_to_r2(self, sid, tag):
        """CPU phase: after a matched short is downloaded, sync + delete local."""
        import os
        import subprocess

        if not os.environ.get("R2_ACCESS_KEY"):
            return

        bucket = "r2:clipwhy-data"

        for subdir_name, raw_subdir in (
            ("shorts_audio", config.RAW_SHORTS_AUDIO_DIR),
            ("shorts_video", config.RAW_SHORTS_VIDEO_DIR),
        ):
            for ext in ("wav", "mp4"):
                f = raw_subdir / f"{sid}.{ext}"
                if not f.exists():
                    continue
                try:
                    subprocess.run(
                        ["rclone", "copyto", str(f),
                         f"{bucket}/raw/{subdir_name}/{sid}.{ext}",
                         "--disable-http2", "--retries", "3",
                         "--update"],
                        capture_output=True, timeout=180, check=False,
                    )
                    f.unlink(missing_ok=True)
                except Exception as e:
                    log.warning("%s R2 sync short %s.%s failed: %s", tag, sid, ext, e)

    # ── Step implementations ────────────────────────────────────────────────

    def _step1_list(self, creator_id, channel_id, tag):
        """List shorts and longs from the channel's uploads playlist."""
        # Check step cache
        cached = progress.load_step_cache(creator_id, "step1_list")
        if cached:
            log.info("%s Step 1: LIST (cached)", tag)
            return cached["short_ids"], cached["long_ids"]

        playlist_id = f"UU{channel_id[2:]}"
        video_ids = self.api.list_playlist_videos(
            playlist_id, max_videos=300
        )
        raw_items = self.api.get_video_details(video_ids)

        # Normalize raw API items to flat dicts
        videos = [config.normalize_video_item(item) for item in raw_items]

        # Classify by duration
        min_age = datetime.now(timezone.utc) - timedelta(days=config.MIN_SHORT_AGE_DAYS)
        short_ids = []
        long_ids = []

        for v in videos:
            dur = v["duration_seconds"]
            if dur <= config.MAX_SHORT_DURATION_SEC:
                if v["published_at"]:
                    pub = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
                    if pub <= min_age:
                        short_ids.append(v["video_id"])
            elif dur >= config.MIN_LONG_VIDEO_DURATION_SEC:
                long_ids.append(v["video_id"])

        short_ids = short_ids[:config.MAX_SHORTS_TO_LIST]
        long_ids = long_ids[:config.MAX_LONGS_TO_LIST]

        progress.save_step_cache(creator_id, "step1_list", {
            "short_ids": short_ids, "long_ids": long_ids,
        })

        log.info("%s Step 1: LIST - %d shorts, %d longs",
                 tag, len(short_ids), len(long_ids))
        return short_ids, long_ids

    def _step2_caption_match(self, creator_id, short_ids, long_ids, tag):
        """Download captions and fuzzy-match shorts to longs."""
        cached = progress.load_step_cache(creator_id, "step2_captions")
        if cached:
            log.info("%s Step 2: CAPTIONS (cached, %d pairs)", tag, len(cached))
            return cached

        pairs = find_caption_pairs(short_ids, long_ids, worker_id=self.worker_id)

        progress.save_step_cache(creator_id, "step2_captions", pairs)

        matched_longs = len(get_matched_long_ids(pairs))
        avg_score = (
            sum(p["caption_score"] for p in pairs) / len(pairs)
            if pairs else 0
        )

        log.info("%s Step 2: CAPTION MATCH - %d longs matched, %d pairs (avg %.1f%%)",
                 tag, matched_longs, len(pairs), avg_score)
        return pairs

    def _step3_metadata(self, creator_id, all_short_ids, comments_disabled, tag):
        """Fetch engagement metadata for ALL shorts early, before deletions."""
        cached = progress.load_step_cache(creator_id, "step3_metadata")
        if cached:
            log.info("%s Step 3: METADATA (cached)", tag)
            shorts_df = pd.read_csv(
                config.METADATA_DIR / f"{creator_id}_shorts.csv"
            )
            baselines = cached.get("baselines", {})
            return shorts_df, baselines

        shorts_df = fetch_shorts_metadata(
            self.api, all_short_ids, creator_id, comments_disabled
        )

        # Save immediately
        shorts_path = config.METADATA_DIR / f"{creator_id}_shorts.csv"
        shorts_path.parent.mkdir(parents=True, exist_ok=True)
        shorts_df.to_csv(shorts_path, index=False)

        # Compute baselines
        baselines = compute_baselines(shorts_df, creator_id, comments_disabled)
        if baselines:
            bl_df = pd.DataFrame([baselines])
            bl_path = config.METADATA_DIR / f"{creator_id}_baselines.csv"
            bl_df.to_csv(bl_path, index=False)

        progress.save_step_cache(creator_id, "step3_metadata", {
            "baselines": baselines,
        })

        eligible = len(shorts_df[shorts_df["excluded_reason"] == ""])
        med_er = baselines.get("median_engagement_rate", 0)
        med_vpd = baselines.get("median_views_per_day", 0)

        log.info("%s Step 3: METADATA - %d shorts, %d eligible, median ER=%.4f, VPD=%.1f",
                 tag, len(shorts_df), eligible, med_er, med_vpd)

        return shorts_df, baselines

    def _step4_download(self, creator_id, matched_long_ids, matched_short_ids, tag):
        """Download matched longs + shorts, segment longs into 30s chunks.

        R2 uploads run async in a per-worker background thread pool so
        downloads do not block on network upload. Step returns after all
        pending uploads complete, so the creator's data is fully on R2 by
        the time the done marker is written.
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        video_segments = {}  # video_id -> {num_segments, duration}
        download_failures = 0
        total_segments = 0

        log.info("%s Step 4: DOWNLOAD - %d longs + %d shorts",
                 tag, len(matched_long_ids), len(matched_short_ids))

        # Async upload pool: 2 parallel rclone jobs per worker. Enough to keep
        # the network saturated without creating too many concurrent uploads
        # per pod when 4 workers each have their own pool.
        use_async_upload = bool(os.environ.get("R2_ACCESS_KEY"))
        upload_pool = ThreadPoolExecutor(max_workers=2) if use_async_upload else None
        upload_futures = []

        # Download and segment matched long videos. Upload each asynchronously.
        for vid in matched_long_ids:
            audio_path = download_long_audio(vid, worker_id=self.worker_id)
            if not audio_path:
                download_failures += 1
                log.warning("%s Long audio download failed: %s", tag, vid)
                continue

            video_path = download_long_video(vid, worker_id=self.worker_id)
            if not video_path:
                log.warning("%s Long video download failed (continuing with audio): %s", tag, vid)

            # Segment
            seg_info = segment_long_video(vid, audio_path, video_path)
            video_segments[vid] = seg_info
            total_segments += seg_info["num_segments"]

            # Fire the upload asynchronously, free to start next download immediately
            if upload_pool is not None:
                upload_futures.append(
                    upload_pool.submit(self._stream_video_to_r2, vid, tag)
                )

        # Download matched shorts (audio + video), same async-upload pattern
        short_failures = 0
        for sid in matched_short_ids:
            if not download_short_audio(sid, worker_id=self.worker_id):
                short_failures += 1
                continue
            download_short_video(sid, worker_id=self.worker_id)  # best-effort
            if upload_pool is not None:
                upload_futures.append(
                    upload_pool.submit(self._stream_short_to_r2, sid, tag)
                )

        # Wait for all background uploads before returning.
        # This guarantees: by the time the creator is marked done, all its
        # audio/video/segment data is on R2 (so GPU phase can pull it cleanly).
        if upload_pool is not None:
            log.info("%s Step 4: waiting for %d background uploads to finish",
                     tag, len(upload_futures))
            for f in upload_futures:
                try:
                    f.result(timeout=1800)  # 30 min safety
                except Exception as e:
                    log.warning("%s async upload failed: %s", tag, e)
            upload_pool.shutdown(wait=True)

        log.info("%s Step 4: DOWNLOAD done - %d longs, %d failed, %d segments, %d/%d shorts",
                 tag, len(video_segments), download_failures, total_segments,
                 len(matched_short_ids) - short_failures, len(matched_short_ids))

        return video_segments, {
            "longs_downloaded": len(video_segments),
            "download_failures": download_failures,
            "total_segments": total_segments,
        }

    def _step5_whisper(self, creator_id, video_segments, matched_short_ids,
                       caption_pairs, tag):
        """Whisper transcription + exact segment matching. Uses GPU lock."""
        log.info("%s Step 5: WHISPER - waiting for GPU lock", tag)

        with self.gpu_lock:
            gpu_start = time.time()
            log.info("%s Step 5: WHISPER - GPU acquired", tag)

            # Transcribe all segments for all matched longs
            all_transcripts = {}  # video_id -> {seg_idx: text}
            total_transcribed = 0

            for vid, seg_info in video_segments.items():
                transcripts = transcribe_segments(
                    self.whisper_model, vid, seg_info["num_segments"]
                )
                all_transcripts[vid] = transcripts
                total_transcribed += len(transcripts)

            # Transcribe all matched shorts
            short_transcripts = {}  # short_id -> text
            for sid in matched_short_ids:
                text = transcribe_short(self.whisper_model, sid)
                if text:
                    short_transcripts[sid] = text

            gpu_time = time.time() - gpu_start

        # GPU released - now do matching (CPU-only, no lock needed)
        # Build per-creator segment transcript pool
        # For each short, match against segments of its caption-matched long
        caption_pairs_by_short = {p["short_id"]: p for p in caption_pairs}

        whisper_pairs = []
        for sid, short_text in short_transcripts.items():
            cap_pair = caption_pairs_by_short.get(sid)
            if not cap_pair:
                continue

            long_id = cap_pair["long_id"]
            seg_transcripts = all_transcripts.get(long_id, {})
            if not seg_transcripts:
                continue

            matches = whisper_match_short_to_segments(short_text, seg_transcripts)
            if matches:
                best_idx, best_score, method = max(matches, key=lambda m: m[1])
                whisper_pairs.append({
                    "short_id": sid,
                    "long_id": long_id,
                    "matched_segment_index": best_idx,
                    "match_score": round(best_score, 1),
                    "match_method": f"whisper_{method}",
                    "match_confidence": (
                        "very_high" if best_score >= config.HIGH_CONFIDENCE_THRESHOLD
                        else "high"
                    ),
                })

        # Save whisper pairs cache
        progress.save_step_cache(creator_id, "step4_whisper", whisper_pairs)

        match_rate = (
            len(whisper_pairs) / len(short_transcripts) * 100
            if short_transcripts else 0
        )
        avg_score = (
            sum(p["match_score"] for p in whisper_pairs) / len(whisper_pairs)
            if whisper_pairs else 0
        )

        log.info("%s Step 5: WHISPER done - %d segs + %d shorts transcribed, "
                 "%d matches (%.0f%%), avg score %.1f%%, GPU %.0f min",
                 tag, total_transcribed, len(short_transcripts),
                 len(whisper_pairs), match_rate, avg_score, gpu_time / 60)

        return whisper_pairs, {
            "segments_transcribed": total_transcribed,
            "shorts_transcribed": len(short_transcripts),
            "whisper_matches": len(whisper_pairs),
            "gpu_time_min": round(gpu_time / 60, 1),
        }

    def _step6_labels(self, creator_id, category, whisper_pairs,
                      shorts_df, baselines, video_segments, tag):
        """Label segments using metadata from step 3 + Whisper matches from step 5."""
        # Label matched shorts (viral or not) using pre-fetched metadata
        labeled_pairs = label_matched_shorts(
            whisper_pairs, shorts_df, baselines
        )

        # Save labeled pairs
        if labeled_pairs:
            pairs_df = pd.DataFrame(labeled_pairs)
            pairs_path = config.PAIRS_DIR / f"{creator_id}_whisper_pairs.csv"
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            pairs_df.to_csv(pairs_path, index=False)

        # Load all segment transcripts for labeling
        segment_transcripts = {}
        for vid in video_segments:
            segment_transcripts[vid] = load_segment_transcripts(vid)

        # Label all segments
        segments_df = label_creator_segments(
            creator_id, category, video_segments, labeled_pairs,
            segment_transcripts
        )

        # Save labeled segments
        seg_path = config.LABELED_DIR / f"{creator_id}_segments.csv"
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        segments_df.to_csv(seg_path, index=False)

        total_segs = len(segments_df)
        positive_segs = int(segments_df["label"].sum()) if total_segs > 0 else 0
        viral_shorts = sum(1 for p in labeled_pairs if p.get("label") == 1)

        log.info("%s Step 6: LABELS - %d viral/%d pairs, %d segments, %d positive",
                 tag, viral_shorts, len(labeled_pairs), total_segs, positive_segs)

        return {
            "total_segments": total_segs,
            "positive_segments": positive_segs,
            "viral_shorts": viral_shorts,
            "total_pairs": len(labeled_pairs),
        }
