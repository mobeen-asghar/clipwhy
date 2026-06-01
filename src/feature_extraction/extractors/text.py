"""Text features (10).

All 10 are pure-Python computations on the Whisper transcript. Two of them
(`first_5s_hook_word_ratio`, `articulation_rate`) use word-level timestamps
from the transcript JSON; the others use the `transcript_text` field in
the labeled CSV directly.
"""
import json
import logging
import math
import re
from typing import Optional

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.text")

_PUNCT_STRIP = str.maketrans("", "", ".,!?;:\"'()[]{}")


def _clean_words(text: str) -> list[str]:
    return [w.translate(_PUNCT_STRIP).lower() for w in text.split() if w.strip()]


def _sentences(text: str) -> list[str]:
    # Split on sentence-ending punctuation; drop fragments < 2 words.
    parts = re.split(r"[.!?]+", text or "")
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


def _load_transcript_words(job: SegmentJob) -> list[dict]:
    """Load word-level timestamps from transcript JSON. Returns [] on any error.

    Logs at WARNING level if the file is missing/corrupt: silent feature
    zero-fills are how training data gets quietly polluted, so make noise.
    """
    try:
        with open(job.transcript_path) as f:
            data = json.load(f)
        return data.get("words", []) or []
    except FileNotFoundError:
        log.warning(
            "transcript JSON missing for %s (%s); first_5s_hook + articulation_rate will be 0",
            job.segment_id, job.transcript_path,
        )
        return []
    except Exception as e:
        log.warning("transcript JSON read failed for %s: %s", job.segment_id, e)
        return []


def extract(job: SegmentJob) -> dict:
    text = job.transcript_text or ""
    words = _clean_words(text)
    sents = _sentences(text)
    n_words = len(words)
    n_sents = len(sents)

    # ── Basic counts ────────────────────────────────────────────────────────
    word_count = n_words
    duration = max(job.duration, 1e-6)
    words_per_second = round(word_count / duration, 4)

    hook_word_count = sum(1 for w in words if w in config.HOOK_WORDS)
    hook_word_ratio = round(hook_word_count / word_count, 6) if word_count else 0.0

    # ── Questions ───────────────────────────────────────────────────────────
    explicit_q = text.count("?")
    implicit_q = 0
    for s in sents:
        ws = s.split()
        if not ws:
            continue
        first = ws[0].translate(_PUNCT_STRIP).lower()
        if first in config.INTERROGATIVE_WORDS and "?" not in s:
            implicit_q += 1
    question_count = explicit_q + implicit_q
    question_density = round(question_count / n_sents, 4) if n_sents else 0.0

    # ── Pronouns ────────────────────────────────────────────────────────────
    second_person_ratio = round(
        sum(1 for w in words if w in config.SECOND_PERSON_WORDS) / word_count,
        6,
    ) if word_count else 0.0
    first_person_ratio = round(
        sum(1 for w in words if w in config.FIRST_PERSON_WORDS) / word_count,
        6,
    ) if word_count else 0.0

    # ── Timestamp-based features ────────────────────────────────────────────
    wlist = _load_transcript_words(job)

    first_5s_hook_word_ratio = _first_5s_hook_ratio(wlist)
    articulation_rate = _articulation_rate(wlist)

    return {
        "word_count": float(word_count),
        "words_per_second": float(words_per_second),
        "hook_word_count": float(hook_word_count),
        "hook_word_ratio": float(hook_word_ratio),
        "question_count": float(question_count),
        "question_density": float(question_density),
        "second_person_ratio": float(second_person_ratio),
        "first_person_ratio": float(first_person_ratio),
        "first_5s_hook_word_ratio": float(first_5s_hook_word_ratio),
        "articulation_rate": float(articulation_rate),
    }


def _first_5s_hook_ratio(wlist: list[dict]) -> float:
    """count(hook_words where end <= 5s) / count(words where end <= 5s)."""
    if not wlist:
        return 0.0
    win = config.OPENING_HOOK_WINDOW_SEC
    opening_words = [w for w in wlist if float(w.get("end", 0)) <= win]
    if not opening_words:
        return 0.0
    cleaned = [
        str(w.get("word", "")).translate(_PUNCT_STRIP).lower()
        for w in opening_words
    ]
    n_total = sum(1 for w in cleaned if w)
    if n_total == 0:
        return 0.0
    n_hook = sum(1 for w in cleaned if w in config.HOOK_WORDS)
    return round(n_hook / n_total, 6)


def _articulation_rate(wlist: list[dict]) -> float:
    """len(words) / (total_span_s - sum(pauses > PAUSE_THRESHOLD_SEC))."""
    if len(wlist) < 2:
        return 0.0
    try:
        starts = [float(w.get("start", 0)) for w in wlist]
        ends = [float(w.get("end", 0)) for w in wlist]
    except Exception:
        return 0.0
    total_span = ends[-1] - starts[0]
    if total_span <= 0:
        return 0.0
    pauses = [
        starts[i + 1] - ends[i]
        for i in range(len(wlist) - 1)
        if starts[i + 1] - ends[i] > config.PAUSE_THRESHOLD_SEC
    ]
    speaking_time = total_span - sum(pauses)
    if speaking_time <= 0:
        return 0.0
    return round(len(wlist) / speaking_time, 4)
