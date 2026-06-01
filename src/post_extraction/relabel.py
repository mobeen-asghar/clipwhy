"""Recompute virality labels at V1's strict thresholds (2.0x ER / 3.0x VPD).

V2's data collection shipped with the V1 pre-adjustment defaults
(ENGAGEMENT_MULTIPLIER=1.5, VIEWS_PER_DAY_MULTIPLIER=2.0). V1's final labels
used 2.0x / 3.0x after auto-adjusting to hit the 30-70% viral split target
(see clipwhy-scraper/compute_labels.py and README Step 4).

Every V2 pair CSV already carries the raw engagement numbers and creator
medians, so no pipeline rerun is needed. We recompute viral status per pair,
build a (video_id, segment_index) -> new_label lookup, and apply it to the
merged features.csv.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("clipwhy.post_extraction.relabel")

DEFAULT_ENG_MULT = 2.0
DEFAULT_VPD_MULT = 3.0


def _load_pairs() -> pd.DataFrame:
    pairs_dir = config.R2_CACHE / "pairs"
    files = sorted(pairs_dir.glob("CR*_whisper_pairs.csv"))
    if not files:
        raise FileNotFoundError(f"No pairs CSVs in {pairs_dir}. Run `download` first.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        # Tag with creator_id derived from filename so we can audit per-creator
        df["creator_id"] = f.stem.split("_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def relabel(
    eng_mult: float = DEFAULT_ENG_MULT,
    vpd_mult: float = DEFAULT_VPD_MULT,
    features_csv: Path | None = None,
) -> pd.DataFrame:
    features_csv = features_csv or config.MERGED_FEATURES_CSV
    if not features_csv.exists():
        raise FileNotFoundError(
            f"{features_csv} not found. Run `merge` first."
        )

    pairs = _load_pairs()
    log.info("Loaded %d total Whisper pairs across %d creators",
             len(pairs), pairs["creator_id"].nunique())

    # Recompute viral status per pair under new thresholds
    median_er = pairs["creator_median_engagement_rate"]
    median_vpd = pairs["creator_median_views_per_day"]
    meets_er = (median_er > 0) & (pairs["short_engagement_rate"] > eng_mult * median_er)
    meets_vpd = (median_vpd > 0) & (pairs["short_views_per_day"] > vpd_mult * median_vpd)
    pairs["new_meets_engagement_threshold"] = meets_er
    pairs["new_meets_views_threshold"] = meets_vpd
    pairs["new_label"] = (meets_er | meets_vpd).astype(int)

    # Viral split audit
    total_pairs = len(pairs)
    old_viral = int(pairs["label"].sum())
    new_viral = int(pairs["new_label"].sum())
    log.info("Pair virality at OLD thresholds (1.5x/2.0x): %d / %d = %.1f%%",
             old_viral, total_pairs, 100 * old_viral / total_pairs)
    log.info("Pair virality at NEW thresholds (%.1fx/%.1fx): %d / %d = %.1f%%",
             eng_mult, vpd_mult, new_viral, total_pairs, 100 * new_viral / total_pairs)

    split_pct = new_viral / total_pairs
    if split_pct < 0.30 or split_pct > 0.70:
        log.warning(
            "Viral split %.1f%% outside V1's 30-70%% target -- consider adjusting",
            100 * split_pct,
        )

    # Build lookup: (long_id, matched_segment_index) -> new_label
    # If a segment is matched by multiple shorts, it's positive if ANY match is viral
    positive_keys: set[tuple[str, int]] = set()
    match_keys: set[tuple[str, int]] = set()
    for long_id, seg_idx, new_label in zip(
        pairs["long_id"], pairs["matched_segment_index"], pairs["new_label"]
    ):
        key = (long_id, int(seg_idx))
        match_keys.add(key)
        if new_label == 1:
            positive_keys.add(key)

    log.info("Unique matched (video, segment) pairs: %d", len(match_keys))
    log.info("Unique positive (video, segment) pairs: %d", len(positive_keys))

    # Apply to features
    features = pd.read_csv(features_csv)
    old_positive_segments = int(features["label"].sum())
    new_labels = [
        1 if (vid, int(idx)) in positive_keys else 0
        for vid, idx in zip(features["video_id"], features["segment_index"])
    ]
    features["label"] = new_labels
    new_positive_segments = int(features["label"].sum())

    log.info("Feature segments: %d positives at OLD thresholds", old_positive_segments)
    log.info("Feature segments: %d positives at NEW thresholds", new_positive_segments)
    log.info("Net change: %+d positive segments (%.2f%% -> %.2f%%)",
             new_positive_segments - old_positive_segments,
             100 * old_positive_segments / len(features),
             100 * new_positive_segments / len(features))

    # Per-category breakdown
    by_cat = features.groupby("category")["label"].agg(["sum", "count"])
    by_cat["pct"] = 100 * by_cat["sum"] / by_cat["count"]
    log.info("Per-category positives (new labels):\n%s", by_cat.to_string())

    # Overwrite features CSV (destination is local only, R2 untouched)
    features.to_csv(features_csv, index=False)
    log.info("Overwrote %s with new labels", features_csv)
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    relabel()
