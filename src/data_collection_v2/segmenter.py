"""
FFmpeg segmentation: split long videos into 30-second chunks.

Adapted from V1 clipwhy-scraper/download_segment.py:
  - segment_audio()
  - segment_video()
  - extract_thumbnails()

Produces per-segment:
  - audio/segment_NNN.wav  (16kHz mono, for Whisper)
  - video/segment_NNN.mp4  (stream copy, for feature extraction)
  - thumbnails/segment_NNN.jpg (first frame, for web UI)
"""

import logging
import math
import subprocess
from pathlib import Path

from .downloader import get_duration
from . import config

log = logging.getLogger("clipwhy.segment")


def segment_audio(audio_path: Path, output_dir: Path,
                  segment_sec: int = None) -> int:
    """Split audio into N-second WAV segments.
    Returns number of segments created."""
    segment_sec = segment_sec or config.SEGMENT_DURATION_SEC
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration(audio_path)
    if duration <= 0:
        return 0

    num_segments = math.ceil(duration / segment_sec)

    for i in range(num_segments):
        start = i * segment_sec
        out = output_dir / f"segment_{i:03d}.wav"
        if out.exists() and out.stat().st_size > 100:
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(audio_path),
                 "-ss", str(start), "-t", str(segment_sec),
                 "-ar", "16000", "-ac", "1",
                 str(out)],
                capture_output=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.warning("Audio segment %d failed: %s", i, e)

    return num_segments


def segment_video(video_path: Path, output_dir: Path,
                  segment_sec: int = None) -> int:
    """Split video into N-second MP4 segments (stream copy, no re-encode).
    Returns number of segments created."""
    segment_sec = segment_sec or config.SEGMENT_DURATION_SEC
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration(video_path)
    if duration <= 0:
        return 0

    num_segments = math.ceil(duration / segment_sec)

    for i in range(num_segments):
        start = i * segment_sec
        out = output_dir / f"segment_{i:03d}.mp4"
        if out.exists() and out.stat().st_size > 100:
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(video_path),
                 "-ss", str(start), "-t", str(segment_sec),
                 "-c", "copy",
                 str(out)],
                capture_output=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.warning("Video segment %d failed: %s", i, e)

    return num_segments


def extract_thumbnails(video_path: Path, output_dir: Path,
                       segment_sec: int = None):
    """Extract first frame of each segment as JPG thumbnail."""
    segment_sec = segment_sec or config.SEGMENT_DURATION_SEC
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration(video_path)
    if duration <= 0:
        return

    num_segments = math.ceil(duration / segment_sec)

    for i in range(num_segments):
        start = i * segment_sec
        out = output_dir / f"segment_{i:03d}.jpg"
        if out.exists() and out.stat().st_size > 100:
            continue
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", str(start), "-i", str(video_path),
                 "-vframes", "1", "-q:v", "2",
                 str(out)],
                capture_output=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.warning("Thumbnail %d failed: %s", i, e)


def segment_long_video(video_id: str, audio_path: Path,
                       video_path: Path | None) -> dict:
    """Segment a long video into 30s chunks (audio + video + thumbnails).

    Returns dict with segment counts and paths:
        {
            "video_id": str,
            "num_segments": int,
            "duration": float,
            "audio_dir": str,
            "video_dir": str | None,
            "thumbnail_dir": str | None,
        }
    """
    seg_audio_dir = config.SEGMENTS_DIR / video_id / "audio"
    seg_video_dir = config.SEGMENTS_DIR / video_id / "video"
    seg_thumb_dir = config.SEGMENTS_DIR / video_id / "thumbnails"

    # Get actual duration from audio file
    duration = get_duration(audio_path)
    if duration <= 0:
        log.warning("Could not get duration for %s", video_id)
        return {"video_id": video_id, "num_segments": 0, "duration": 0}

    # Segment audio (always)
    num_segments = segment_audio(audio_path, seg_audio_dir)

    # Segment video + thumbnails (if video was downloaded)
    if video_path and video_path.exists():
        segment_video(video_path, seg_video_dir)
        extract_thumbnails(video_path, seg_thumb_dir)

    log.info("  Segmented %s: %d segments (%.0f min)",
             video_id, num_segments, duration / 60)

    return {
        "video_id": video_id,
        "num_segments": num_segments,
        "duration": duration,
        "audio_dir": str(seg_audio_dir),
        "video_dir": str(seg_video_dir) if video_path else None,
        "thumbnail_dir": str(seg_thumb_dir) if video_path else None,
    }
