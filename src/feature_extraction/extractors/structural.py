"""Structural features (7 per-segment + 1 novelty added post-hoc).

Per-segment features come directly from labeled CSV columns; no media I/O.
segment_novelty_to_neighbors is added by novelty.apply() after all CLIP
embeddings for the creator have been computed.
"""
from ..pipeline import SegmentJob


def extract(job: SegmentJob) -> dict:
    video_dur = max(job.video_duration, 1e-6)
    position_ratio = job.start_time / video_dur
    return {
        "position_ratio": round(position_ratio, 4),
        "is_intro": 1.0 if position_ratio < 0.10 else 0.0,
        "is_outro": 1.0 if position_ratio > 0.90 else 0.0,
        "segment_duration": round(job.duration, 2),
        "video_duration": round(job.video_duration, 2),
        "is_first_segment": 1.0 if job.segment_index == 0 else 0.0,
        "is_last_segment": 1.0 if job.segment_index == job.max_segment_index_in_video else 0.0,
        # segment_novelty_to_neighbors is filled by novelty.apply()
        "segment_novelty_to_neighbors": 0.0,
    }
