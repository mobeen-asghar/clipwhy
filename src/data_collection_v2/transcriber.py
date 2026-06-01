"""
Whisper GPU transcription for segments and shorts.

Adapted from V1 clipwhy-scraper/rematch_whisper.py:
  - transcribe_short() generalized to handle both shorts and segments

The Whisper model is loaded ONCE per VM and shared across threads.
Only one thread uses it at a time (via GPULock).
"""

import json
import logging
import math
from pathlib import Path

from . import config

log = logging.getLogger("clipwhy.transcribe")


def load_whisper_model():
    """Load Whisper model onto GPU (or CPU fallback).
    Called once per VM at startup."""
    import whisper
    import torch

    device = config.WHISPER_DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    log.info("Loading Whisper '%s' on %s...", config.WHISPER_MODEL_NAME, device)
    model = whisper.load_model(config.WHISPER_MODEL_NAME, device=device)
    log.info("Whisper model loaded")
    return model


def transcribe_audio(whisper_model, audio_path: Path, output_path: Path) -> str:
    """Transcribe an audio file with Whisper. Save JSON with text + word timestamps.

    Returns transcript text, or empty string on failure.
    Skips if output JSON already exists (cache).
    """
    # Check cache
    if output_path.exists() and output_path.stat().st_size > 10:
        try:
            data = json.loads(output_path.read_text())
            return data.get("text", "")
        except (json.JSONDecodeError, OSError):
            pass

    if not audio_path.exists():
        return ""

    try:
        result = whisper_model.transcribe(
            str(audio_path), language="en", word_timestamps=True
        )
        text = result["text"].strip()

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"text": text, "words": words}, f)

        return text
    except Exception as e:
        log.warning("Transcription failed for %s: %s", audio_path.name, e)
        return ""


def transcribe_segments(
    whisper_model, video_id: str, num_segments: int
) -> dict[int, str]:
    """Transcribe all 30s segments for a long video.

    Returns dict: {segment_index: transcript_text}
    """
    seg_audio_dir = config.SEGMENTS_DIR / video_id / "audio"
    transcript_dir = config.TRANSCRIPTS_SEGMENTS_DIR / video_id

    transcripts = {}
    for i in range(num_segments):
        audio_path = seg_audio_dir / f"segment_{i:03d}.wav"
        output_path = transcript_dir / f"segment_{i:03d}.json"
        text = transcribe_audio(whisper_model, audio_path, output_path)
        if text:
            transcripts[i] = text

    return transcripts


def transcribe_short(whisper_model, short_id: str) -> str:
    """Transcribe a short's audio. Returns text or empty string."""
    audio_path = config.RAW_SHORTS_AUDIO_DIR / f"{short_id}.wav"
    output_path = config.TRANSCRIPTS_SHORTS_DIR / f"{short_id}.json"
    return transcribe_audio(whisper_model, audio_path, output_path)


def load_segment_transcripts(video_id: str) -> dict[int, str]:
    """Load existing Whisper transcripts for a long video's segments.
    Returns dict: {segment_index: transcript_text}"""
    transcript_dir = config.TRANSCRIPTS_SEGMENTS_DIR / video_id
    if not transcript_dir.exists():
        return {}

    transcripts = {}
    for json_file in sorted(transcript_dir.glob("segment_*.json")):
        try:
            seg_idx = int(json_file.stem.split("_")[1])
            data = json.loads(json_file.read_text())
            text = data.get("text", "").strip()
            if text:
                transcripts[seg_idx] = text
        except (ValueError, json.JSONDecodeError, OSError):
            continue
    return transcripts


def count_transcribed_segments(video_id: str) -> int:
    """Count how many segment transcripts exist for a video (for resume)."""
    transcript_dir = config.TRANSCRIPTS_SEGMENTS_DIR / video_id
    if not transcript_dir.exists():
        return 0
    return len(list(transcript_dir.glob("segment_*.json")))
