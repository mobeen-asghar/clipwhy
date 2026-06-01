"""
VTT caption parsing and timed chunk building.

Adapted from V1 clipwhy-scraper/find_pairs.py:
  - parse_vtt_plain()
  - parse_vtt_timed()
  - build_timed_chunks()
  - parse_timestamp()
"""

import re
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    """Parse VTT timestamp '00:01:23.456' or '01:23.456' to seconds."""
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return 0.0


def parse_vtt_plain(path: Path) -> str:
    """Extract plain text from VTT, deduplicated.
    Used for full-text fuzzy matching."""
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

    # Deduplicate consecutive identical lines (VTT caption jitter)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)
    return " ".join(deduped)


def parse_vtt_timed(path: Path) -> list[tuple[float, str]]:
    """Parse VTT into list of (start_sec, text) entries.
    Used for timed chunk building."""
    entries = []
    text = path.read_text(errors="replace")
    blocks = re.split(r"\n\n+", text)

    for block in blocks:
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            match = re.match(
                r"(\d{2}:\d{2}[\d:.]+)\s*-->\s*(\d{2}:\d{2}[\d:.]+)", line
            )
            if match:
                start = parse_timestamp(match.group(1))
                text_lines = []
                for tl in lines[i + 1:]:
                    tl = re.sub(r"<[^>]+>", "", tl.strip())
                    if tl:
                        text_lines.append(tl)
                if text_lines:
                    entries.append((start, " ".join(text_lines)))
                break

    return entries


def build_timed_chunks(
    entries: list[tuple[float, str]], chunk_sec: int = 30
) -> list[tuple[float, float, str]]:
    """Group VTT entries into N-second time windows.
    Returns list of (start_sec, end_sec, chunk_text).
    Text within each chunk is deduplicated (handles VTT overlap jitter)."""
    if not entries:
        return []

    chunks = []
    chunk_start = entries[0][0]
    chunk_texts = []

    for start, text in entries:
        if start - chunk_start >= chunk_sec and chunk_texts:
            chunks.append((chunk_start, start, " ".join(chunk_texts)))
            chunk_start = start
            chunk_texts = [text]
        else:
            chunk_texts.append(text)

    if chunk_texts:
        chunks.append((
            chunk_start,
            entries[-1][0] + chunk_sec,
            " ".join(chunk_texts),
        ))

    # Deduplicate consecutive repeated words within each chunk
    deduped_chunks = []
    for start, end, text in chunks:
        words = text.split()
        deduped = []
        for w in words:
            if not deduped or w != deduped[-1]:
                deduped.append(w)
        deduped_chunks.append((start, end, " ".join(deduped)))

    return deduped_chunks
