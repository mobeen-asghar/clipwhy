"""Stages 4-5: English verification and repurpose confirmation.

Uses parallel workers to speed up download + processing:
- Stage 4: ThreadPool downloads audio, main thread runs Whisper
- Stage 5: ThreadPool downloads captions, ProcessPool runs fuzzy matching
"""

import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from config.settings import (
    WHISPER_MODEL,
    REPURPOSE_NUM_SHORTS, REPURPOSE_NUM_LONGS,
    REPURPOSE_MATCH_THRESHOLD, REPURPOSE_MIN_MATCHES,
    DOWNLOAD_WORKERS, PREFETCH_BATCH, PROXY_URL,
)

log = logging.getLogger("clipwhy.verify")

YTDLP_BIN = shutil.which("yt-dlp") or "yt-dlp"


def _proxy_args():
    """Return yt-dlp proxy arguments if PROXY_URL is configured."""
    if PROXY_URL:
        return ["--proxy", PROXY_URL]
    return []


def _load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ── Audio download ────────────────────────────────────────────────────────────



def _download_audio_snippet(video_id, audio_dir):
    """Download 30s of audio from a video as 16kHz mono WAV.

    Strategy (in order):
    1. Try --download-sections *2:00-2:30 (fast, avoids English intros)
    2. If that fails (HLS can't seek), try --download-sections *0:00-0:30
    3. If that also fails, download full audio and trim with ffmpeg -ss/-t
    """
    wav = audio_dir / f"{video_id}_30s.wav"
    if wav.exists() and wav.stat().st_size > 1000:
        return wav

    audio_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup():
        for ext in (".mp4", ".mp4.part", ".mp4.ytdl", ".webm", ".webm.part",
                     ".m4a", ".m4a.part"):
            for p in audio_dir.glob(f"{video_id}_30s{ext}"):
                p.unlink(missing_ok=True)

    def _try_download(extra_args, timeout=120):
        _cleanup()
        cmd = [
            YTDLP_BIN, "--no-warnings", "-q", "-x",
            "--audio-format", "wav",
            *_proxy_args(),
            *extra_args,
            "-o", str(wav), "--force-overwrites",
            f"https://youtube.com/watch?v={video_id}",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout)
            return wav.exists() and wav.stat().st_size > 1000
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            return False

    # Attempt 1: section from 2:00-2:30 (avoids English intros)
    if _try_download(["--download-sections", "*2:00-2:30",
                       "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000"]):
        return wav

    # Attempt 2: section from 0:00-0:30 (simpler seek, works on more formats)
    if _try_download(["--download-sections", "*0:00-0:30",
                       "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000"]):
        return wav

    # Attempt 3: download full audio, trim with ffmpeg postprocessor
    # Timeout is longer since it downloads the full file
    if _try_download(["--postprocessor-args",
                       "ffmpeg:-ss 120 -t 30 -ac 1 -ar 16000"], timeout=300):
        return wav

    return None


# ── Captions download + parse ─────────────────────────────────────────────────

def _parse_vtt(path):
    """Parse VTT subtitle file to plain text, deduplicating overlapping lines."""
    lines = []
    for line in path.read_text(errors="replace").split("\n"):
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") \
                or line.startswith("Language:") or line.startswith("NOTE"):
            continue
        if re.match(r"^\d{2}:\d{2}", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return " ".join(deduped)


def _get_captions(video_id, captions_dir):
    """Download YouTube auto-captions via yt-dlp. Returns text or None."""
    vtt = captions_dir / f"{video_id}.en.vtt"
    if vtt.exists():
        return _parse_vtt(vtt) if vtt.stat().st_size > 100 else None

    captions_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        YTDLP_BIN, "--write-auto-sub", "--sub-lang", "en",
        "--skip-download", "--sub-format", "vtt", "--no-warnings", "-q",
        *_proxy_args(),
        "-o", str(captions_dir / "%(id)s"),
        f"https://youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return None

    if vtt.exists() and vtt.stat().st_size > 100:
        return _parse_vtt(vtt)
    return None


# ── Parallel download helpers ─────────────────────────────────────────────────

def _download_audio_for_channel(args):
    """Download audio for one channel. Returns (channel_id, wav_path_or_None)."""
    cid, video_id, audio_dir = args
    wav = _download_audio_snippet(video_id, audio_dir)
    return cid, wav


def _download_captions_for_channel(args):
    """Download all captions for one channel. Returns (cid, short_caps, long_caps)."""
    cid, short_ids, long_ids, captions_dir = args

    short_captions = []
    for vid in short_ids:
        text = _get_captions(vid, captions_dir)
        if text and len(text) >= 20:
            short_captions.append(text)

    long_captions = []
    for vid in long_ids:
        text = _get_captions(vid, captions_dir)
        if text and len(text) >= 20:
            long_captions.append(text)

    return cid, short_captions, long_captions


def _fuzzy_match_channel(short_captions, long_captions, threshold):
    """Fuzzy match Shorts against Longs. Returns match count."""
    matches = 0
    for short_text in short_captions:
        short_lower = short_text.lower()
        matched = False
        for lt in long_captions:
            lt_lower = lt.lower()
            if len(lt_lower) <= 3000:
                score = fuzz.partial_ratio(short_lower, lt_lower)
            else:
                score = 0
                for i in range(0, len(lt_lower), 1500):
                    chunk = lt_lower[i:i + 2000]
                    s = fuzz.partial_ratio(short_lower, chunk)
                    if s > score:
                        score = s
                    if score >= threshold:
                        break
            if score >= threshold:
                matched = True
                break
        if matched:
            matches += 1
    return matches


# ── Video IDs helper ──────────────────────────────────────────────────────────

def _load_video_ids(cache_dir):
    """Load video IDs from JSON cache, falling back to stage3 CSV if needed."""
    cache_path = cache_dir / "video_ids.json"
    vid_cache = _load_json(cache_path)

    if vid_cache:
        return vid_cache

    # Fallback: rebuild from stage3_counts.csv (has short_video_ids, long_video_ids columns)
    csv_path = cache_dir.parent / "stage3_counts.csv"
    if csv_path.exists():
        log.info("Rebuilding video_ids cache from stage3_counts.csv")
        df = pd.read_csv(csv_path)
        if "short_video_ids" in df.columns and "long_video_ids" in df.columns:
            for _, row in df.iterrows():
                cid = row["channel_id"]
                short_str = str(row.get("short_video_ids", ""))
                long_str = str(row.get("long_video_ids", ""))
                vid_cache[cid] = {
                    "short_ids": [s for s in short_str.split(",") if s and s != "nan"],
                    "long_ids": [s for s in long_str.split(",") if s and s != "nan"],
                }
            _save_json(cache_path, vid_cache)
            log.info("Rebuilt video_ids cache: %d channels", len(vid_cache))

    return vid_cache


# ── Stage 4: English verification (parallel downloads) ────────────────────────

def verify_english(channels_df, cache_dir, notify=None):
    """
    Download 30s audio from 1 long video per channel, run Whisper tiny.
    Downloads run in parallel (4 workers), Whisper runs sequentially on main thread.
    """
    whisper_cache_path = cache_dir / "whisper_english_cache.json"
    whisper_cache = _load_json(whisper_cache_path)

    video_ids_cache = _load_video_ids(cache_dir)
    audio_dir = cache_dir / "audio"

    # Separate cached from uncached channels
    verified = []
    to_process = []

    for _, row in channels_df.iterrows():
        cid = row["channel_id"]
        if cid in whisper_cache:
            if whisper_cache[cid].get("english"):
                row_dict = row.to_dict()
                row_dict["whisper_en_prob"] = whisper_cache[cid].get("prob", 0)
                verified.append(row_dict)
            continue

        ids_entry = video_ids_cache.get(cid, {})
        long_ids = ids_entry.get("long_ids", [])
        if not long_ids:
            whisper_cache[cid] = {"english": False, "method": "no_long_videos"}
            continue

        to_process.append((cid, long_ids[0], row.to_dict()))

    if not to_process:
        _save_json(whisper_cache_path, whisper_cache)
        log.info("Stage 4: all %d channels already cached", len(channels_df))
        return pd.DataFrame(verified)

    log.info("Stage 4: %d cached, %d to process (parallel downloads)",
             len(channels_df) - len(to_process), len(to_process))

    # Load Whisper once
    import whisper as _whisper
    log.info("Loading Whisper model: %s", WHISPER_MODEL)
    whisper_model = _whisper.load_model(WHISPER_MODEL)
    log.info("Whisper loaded")

    # Process in batches: pre-download audio in parallel, then run Whisper
    done = 0
    for batch_start in range(0, len(to_process), PREFETCH_BATCH):
        batch = to_process[batch_start:batch_start + PREFETCH_BATCH]

        # Parallel download audio for this batch
        download_args = [(cid, vid, audio_dir) for cid, vid, _ in batch]
        audio_results = {}

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(_download_audio_for_channel, a): a[0]
                       for a in download_args}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    _, wav = future.result()
                    audio_results[cid] = wav
                except Exception as e:
                    log.warning("Download error %s: %s", cid, e)
                    audio_results[cid] = None

        # Run Whisper sequentially on downloaded audio
        for cid, vid, row_dict in batch:
            wav = audio_results.get(cid)
            if not wav:
                whisper_cache[cid] = {"english": False, "method": "download_failed"}
                done += 1
                continue

            try:
                audio = _whisper.load_audio(str(wav))
                audio = _whisper.pad_or_trim(audio)
                mel = _whisper.log_mel_spectrogram(audio).to(whisper_model.device)
                _, probs = whisper_model.detect_language(mel)
                en_prob = probs.get("en", 0)
            except Exception as e:
                log.warning("Whisper error %s: %s", cid, e)
                whisper_cache[cid] = {"english": False, "method": f"error:{str(e)[:40]}"}
                done += 1
                continue

            is_english = en_prob > 0.5
            whisper_cache[cid] = {
                "english": is_english,
                "method": "whisper",
                "prob": round(en_prob, 3),
            }

            if is_english:
                row_dict["whisper_en_prob"] = round(en_prob, 3)
                verified.append(row_dict)

            done += 1

        _save_json(whisper_cache_path, whisper_cache)
        msg = (f"  Stage 4: {done}/{len(to_process)} checked, "
               f"{len(verified)} English")
        log.info(msg)
        if notify:
            notify.send(msg)

    _save_json(whisper_cache_path, whisper_cache)
    log.info("Stage 4 complete: %d/%d English verified",
             len(verified), len(channels_df))
    return pd.DataFrame(verified)


# ── Stage 5: Repurpose verification (parallel downloads + matching) ───────────

def verify_repurpose(channels_df, cache_dir, notify=None):
    """
    For each channel, download captions for 5 Shorts + 5 Longs (same channel).
    Fuzzy match Short captions against Long captions.
    Downloads run in parallel (4 workers), matching is fast after chunking fix.
    No captions = skip channel.
    """
    repurpose_cache_path = cache_dir / "repurpose_cache.json"
    repurpose_cache = _load_json(repurpose_cache_path)

    video_ids_cache = _load_video_ids(cache_dir)
    captions_dir = cache_dir / "captions"

    # Separate cached from uncached
    verified = []
    to_process = []

    for _, row in channels_df.iterrows():
        cid = row["channel_id"]
        if cid in repurpose_cache:
            if repurpose_cache[cid].get("verified"):
                row_dict = row.to_dict()
                row_dict["repurpose_matches"] = repurpose_cache[cid]["matches"]
                verified.append(row_dict)
            continue

        ids_entry = video_ids_cache.get(cid, {})
        short_ids = ids_entry.get("short_ids", [])[:REPURPOSE_NUM_SHORTS]
        long_ids = ids_entry.get("long_ids", [])[:REPURPOSE_NUM_LONGS]

        if len(short_ids) < 2 or len(long_ids) < 2:
            repurpose_cache[cid] = {
                "verified": False, "method": "insufficient_videos", "matches": 0,
            }
            continue

        to_process.append((cid, short_ids, long_ids, row.to_dict()))

    if not to_process:
        _save_json(repurpose_cache_path, repurpose_cache)
        log.info("Stage 5: all %d channels already cached", len(channels_df))
        return pd.DataFrame(verified)

    log.info("Stage 5: %d cached, %d to process (parallel downloads)",
             len(channels_df) - len(to_process), len(to_process))

    # Process in batches: parallel download, then match
    done = 0
    for batch_start in range(0, len(to_process), PREFETCH_BATCH):
        batch = to_process[batch_start:batch_start + PREFETCH_BATCH]

        # Parallel download captions for this batch
        download_args = [(cid, sids, lids, captions_dir)
                         for cid, sids, lids, _ in batch]
        caption_results = {}

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(_download_captions_for_channel, a): a[0]
                       for a in download_args}
            for future in as_completed(futures):
                cid_key = futures[future]
                try:
                    cid, short_caps, long_caps = future.result()
                    caption_results[cid] = (short_caps, long_caps)
                except Exception as e:
                    log.warning("Caption download error %s: %s", cid_key, e)
                    caption_results[cid_key] = ([], [])

        # Run fuzzy matching for this batch
        for cid, short_ids, long_ids, row_dict in batch:
            short_caps, long_caps = caption_results.get(cid, ([], []))

            if not short_caps or not long_caps:
                repurpose_cache[cid] = {
                    "verified": False, "method": "no_captions", "matches": 0,
                }
                done += 1
                continue

            matches = _fuzzy_match_channel(
                short_caps, long_caps, REPURPOSE_MATCH_THRESHOLD
            )

            is_repurposer = matches >= REPURPOSE_MIN_MATCHES
            repurpose_cache[cid] = {
                "verified": is_repurposer,
                "method": "captions",
                "matches": matches,
            }

            if is_repurposer:
                row_dict["repurpose_matches"] = matches
                verified.append(row_dict)

            done += 1

        _save_json(repurpose_cache_path, repurpose_cache)
        msg = (f"  Stage 5: {done}/{len(to_process)} checked, "
               f"{len(verified)} repurposers")
        log.info(msg)
        if notify:
            notify.send(msg)

    _save_json(repurpose_cache_path, repurpose_cache)
    log.info("Stage 5 complete: %d/%d confirmed repurposers",
             len(verified), len(channels_df))
    return pd.DataFrame(verified)
