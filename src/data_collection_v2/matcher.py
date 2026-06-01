"""
Matching logic: caption pre-filter and Whisper segment-level matching.

Two phases:
  1. Caption matching (cheap pre-filter): decides which longs to download
  2. Whisper matching (accurate labels): decides which segments are positive

Adapted from V1:
  - clipwhy-scraper/find_pairs.py (find_best_match, two-phase fuzzy)
  - clipwhy-scraper/rematch_whisper.py (match_short_to_segments)
"""

import logging
from pathlib import Path

from rapidfuzz import fuzz

from . import config
from .captions import parse_vtt_plain, parse_vtt_timed, build_timed_chunks
from .downloader import download_caption

log = logging.getLogger("clipwhy.match")


# ── Phase 1: Caption pre-filter ─────────────────────────────────────────────
# Used in Step 2 to decide which longs to download.
# Threshold is 80 (lower than V1's 90) because this is just a filter.


def caption_match_short_to_long(
    short_text: str, long_full_text: str, long_chunks: list
) -> tuple[float, float, float]:
    """Caption matching using character chunking + timed chunks.

    Phase 1: Match short against 2000-char windows of long text
             (same approach as V2 discovery verify.py).
             partial_ratio on full 20K-90K text gives diluted scores.
             Chunking into 2000-char windows finds the actual match.
    Phase 2: If matched, find approximate timestamp using timed chunks.

    Returns (score, approx_start_sec, approx_end_sec).
    Returns (score, 0, 0) if score below threshold.
    """
    if not short_text or not long_full_text:
        return 0, 0, 0

    short_lower = short_text.lower()
    long_lower = long_full_text.lower()

    # Phase 1: chunked matching (matches discovery's approach exactly)
    if len(long_lower) <= 3000:
        best_score = fuzz.partial_ratio(short_lower, long_lower)
    else:
        best_score = 0
        for i in range(0, len(long_lower), 1500):
            chunk = long_lower[i:i + 2000]
            s = fuzz.partial_ratio(short_lower, chunk)
            if s > best_score:
                best_score = s
            if best_score >= config.CAPTION_MATCH_THRESHOLD:
                break

    if best_score < config.CAPTION_MATCH_THRESHOLD:
        return best_score, 0, 0

    # Phase 2: find approximate timestamp using timed chunks
    best_chunk_score = 0
    best_start = 0.0
    best_end = 0.0

    for start, end, chunk_text in long_chunks:
        if not chunk_text:
            continue
        score = fuzz.partial_ratio(short_lower, chunk_text.lower())
        if score > best_chunk_score:
            best_chunk_score = score
            best_start = start
            best_end = end

    return best_score, best_start, best_end


def find_caption_pairs(
    short_ids: list[str],
    long_ids: list[str],
    worker_id: int = 0,
) -> list[dict]:
    """Match shorts against longs using YouTube captions.

    Downloads captions for all shorts and longs, then fuzzy-matches.
    Returns list of match dicts with short_id, long_id, score, timestamp.
    """
    # Download and parse long video captions
    long_data = []  # (video_id, full_text, chunks)
    failed_downloads = 0
    for lid in long_ids[:config.MAX_LONGS_FOR_CAPTIONS]:
        vtt = download_caption(lid, worker_id=worker_id)
        if not vtt:
            failed_downloads += 1
            continue
        full_text = parse_vtt_plain(vtt)
        entries = parse_vtt_timed(vtt)
        chunks = build_timed_chunks(entries, config.CHUNK_SECONDS)
        if full_text and len(full_text) > 100:
            long_data.append((lid, full_text, chunks))

    checked = min(len(long_ids), config.MAX_LONGS_FOR_CAPTIONS)
    log.info("  %d/%d longs with captions (%d download failures)",
             len(long_data), checked, failed_downloads)
    if not long_data:
        return []

    # Match each short against all longs
    pairs = []
    long_pair_counts = {}  # long_id -> count

    short_downloaded = 0
    short_failed = 0
    for sid in short_ids:
        vtt = download_caption(sid, worker_id=worker_id)
        if not vtt:
            short_failed += 1
            continue
        short_downloaded += 1
        short_text = parse_vtt_plain(vtt)
        if len(short_text) < 20:
            continue

        # Find best matching long
        best_score = 0
        best_long_id = None
        best_start = 0.0
        best_end = 0.0

        for lid, full_text, chunks in long_data:
            if long_pair_counts.get(lid, 0) >= config.MAX_PAIRS_PER_LONG_VIDEO:
                continue
            score, start, end = caption_match_short_to_long(
                short_text, full_text, chunks
            )
            if score > best_score:
                best_score = score
                best_long_id = lid
                best_start = start
                best_end = end

        if best_score >= config.CAPTION_MATCH_THRESHOLD and best_long_id:
            long_pair_counts[best_long_id] = long_pair_counts.get(best_long_id, 0) + 1
            pairs.append({
                "short_id": sid,
                "long_id": best_long_id,
                "caption_score": round(best_score, 1),
                "approx_start": best_start,
                "approx_end": best_end,
            })

    log.info("  %d/%d shorts with captions (%d download failures)",
             short_downloaded, len(short_ids), short_failed)

    return pairs


def get_matched_long_ids(caption_pairs: list[dict]) -> set[str]:
    """Extract unique long video IDs that have at least one matching short."""
    return {p["long_id"] for p in caption_pairs}


def get_matched_short_ids(caption_pairs: list[dict]) -> set[str]:
    """Extract unique short IDs that matched a long video."""
    return {p["short_id"] for p in caption_pairs}


# ── Phase 2: Whisper segment-level matching ─────────────────────────────────
# Used in Step 4 after Whisper transcription for accurate labeling.


def whisper_match_short_to_segments(
    short_text: str,
    segment_transcripts: dict[int, str],
) -> list[tuple[int, float, str]]:
    """Match a short's Whisper transcript against all segment transcripts.

    Tries both single segments and combined adjacent segments
    (for shorts spanning segment boundaries).

    Returns list of (segment_index, score, method) above threshold.
    Method is "single" or "combined".
    """
    if not short_text or len(short_text.split()) < config.MIN_SHORT_WORDS:
        return []

    short_lower = short_text.lower()
    matches = []
    sorted_indices = sorted(segment_transcripts.keys())

    # Single segment matching
    for seg_idx in sorted_indices:
        seg_text = segment_transcripts[seg_idx]
        if not seg_text:
            continue
        score = fuzz.partial_ratio(short_lower, seg_text.lower())
        if score >= config.WHISPER_MATCH_THRESHOLD:
            matches.append((seg_idx, score, "single"))

    # Combined adjacent segments (for shorts spanning 2 segments)
    for i, seg_idx in enumerate(sorted_indices[:-1]):
        next_idx = sorted_indices[i + 1]
        if next_idx != seg_idx + 1:
            continue

        combined = (
            segment_transcripts[seg_idx] + " "
            + segment_transcripts[next_idx]
        )
        score = fuzz.partial_ratio(short_lower, combined.lower())
        if score >= config.WHISPER_MATCH_THRESHOLD:
            # Attribute to the segment with better individual score
            score_a = fuzz.partial_ratio(
                short_lower, segment_transcripts[seg_idx].lower()
            )
            score_b = fuzz.partial_ratio(
                short_lower, segment_transcripts[next_idx].lower()
            )
            best_seg = seg_idx if score_a >= score_b else next_idx
            matches.append((best_seg, score, "combined"))

    # Deduplicate: keep best score per segment
    best_per_seg: dict[int, tuple[float, str]] = {}
    for seg_idx, score, method in matches:
        if seg_idx not in best_per_seg or score > best_per_seg[seg_idx][0]:
            best_per_seg[seg_idx] = (score, method)

    return [
        (idx, score, method)
        for idx, (score, method) in sorted(best_per_seg.items())
    ]
