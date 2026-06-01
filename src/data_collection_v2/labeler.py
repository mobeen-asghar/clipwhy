"""
Segment labeling from Whisper matches.

Key difference from V1: V1 used caption-derived timestamps and overlap
computation. V2 uses direct segment indices from Whisper matching, which
is more accurate (no timestamp approximation).

Adapted from V1 clipwhy-scraper/label_segments.py.
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("clipwhy.label")


def label_creator_segments(
    creator_id: str,
    category: str,
    video_segments: dict[str, dict],
    whisper_pairs: list[dict],
    segment_transcripts: dict[str, dict[int, str]],
) -> pd.DataFrame:
    """Label all segments for a creator's long videos.

    Args:
        creator_id: e.g. "CR0001"
        category: e.g. "tech"
        video_segments: {video_id: {"num_segments": int, "duration": float}}
        whisper_pairs: list of labeled pairs from engagement.label_matched_shorts()
            Each has: short_id, long_id, matched_segment_index, label, etc.
        segment_transcripts: {video_id: {seg_idx: text}}

    Returns DataFrame with 24 columns matching V1 segments.csv schema.
    """
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    segment_sec = config.SEGMENT_DURATION_SEC

    # Build lookup: (long_id, segment_index) -> pair info
    pair_lookup: dict[tuple[str, int], dict] = {}
    for pair in whisper_pairs:
        key = (pair["long_id"], pair["matched_segment_index"])
        # Keep highest-confidence match if multiple shorts match same segment
        if key not in pair_lookup or pair["match_score"] > pair_lookup[key].get("match_score", 0):
            pair_lookup[key] = pair

    rows = []

    for video_id, seg_info in video_segments.items():
        num_segments = seg_info["num_segments"]
        duration = seg_info["duration"]
        if num_segments == 0 or duration <= 0:
            continue

        vid_transcripts = segment_transcripts.get(video_id, {})

        for i in range(num_segments):
            seg_start = i * segment_sec
            seg_end = min(seg_start + segment_sec, duration)
            seg_dur = seg_end - seg_start
            position_ratio = seg_start / duration if duration > 0 else 0

            # Check if this segment has a Whisper match
            pair = pair_lookup.get((video_id, i))

            if pair and pair.get("label") == 1:
                label = 1
                matched_short_id = pair["short_id"]
                matched_short_views = pair.get("short_views", "")
                matched_short_er = pair.get("short_engagement_rate", "")
                virality_raw = matched_short_er
                match_confidence = (
                    "very_high" if pair.get("match_score", 0) >= config.HIGH_CONFIDENCE_THRESHOLD
                    else "high"
                )
            elif pair:
                # Matched but not viral
                label = 0
                matched_short_id = pair["short_id"]
                matched_short_views = pair.get("short_views", "")
                matched_short_er = pair.get("short_engagement_rate", "")
                virality_raw = ""
                match_confidence = "high"
            else:
                label = 0
                matched_short_id = ""
                matched_short_views = ""
                matched_short_er = ""
                virality_raw = ""
                match_confidence = ""

            # Transcript text from Whisper
            transcript = vid_transcripts.get(i, "")

            # Paths (relative to shared storage root)
            seg_audio = f"segments/{video_id}/audio/segment_{i:03d}.wav"
            seg_video = f"segments/{video_id}/video/segment_{i:03d}.mp4"
            seg_thumb = f"segments/{video_id}/thumbnails/segment_{i:03d}.jpg"

            seg_id = f"SEG_{video_id[:8]}_{i:03d}"

            rows.append({
                "segment_id": seg_id,
                "video_id": video_id,
                "creator_id": creator_id,
                "category": category,
                "segment_index": i,
                "start_time": round(seg_start, 1),
                "end_time": round(seg_end, 1),
                "duration": round(seg_dur, 1),
                "video_duration": round(duration, 1),
                "position_ratio": round(position_ratio, 4),
                "is_first_quarter": 1 if position_ratio < 0.25 else 0,
                "is_last_quarter": 1 if position_ratio > 0.75 else 0,
                "audio_path": seg_audio,
                "video_path": seg_video,
                "thumbnail_path": seg_thumb,
                "transcript_text": transcript,
                "label": label,
                "virality_score_raw": virality_raw,
                "matched_short_id": matched_short_id,
                "match_confidence": match_confidence,
                "matched_short_views": matched_short_views,
                "matched_short_engagement_rate": matched_short_er,
                "collected_at": collected_at,
                "split": "",
            })

    return pd.DataFrame(rows)
