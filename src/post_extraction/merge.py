"""Merge per-creator feature CSVs into a single master features.csv.

Reads every CR*_features.csv from data/r2_cache/features/ and writes one
combined CSV to data/post_extraction/features.csv. Column order is enforced
against OUTPUT_COLUMN_ORDER; any creator missing columns or having extras
fails hard so the bug is caught early.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config

log = logging.getLogger("clipwhy.post_extraction.merge")


def merge() -> pd.DataFrame:
    files = sorted(config.R2_FEATURES_DIR.glob("CR*_features.csv"))
    if not files:
        raise FileNotFoundError(
            f"No feature CSVs in {config.R2_FEATURES_DIR}. Run `download` first."
        )

    expected_cols = set(config.OUTPUT_COLUMN_ORDER)
    frames = []
    for f in files:
        df = pd.read_csv(f)
        cols = set(df.columns)
        if cols != expected_cols:
            missing = expected_cols - cols
            extra = cols - expected_cols
            raise ValueError(
                f"{f.name} column mismatch. missing={sorted(missing)[:5]} "
                f"extra={sorted(extra)[:5]}"
            )
        frames.append(df[config.OUTPUT_COLUMN_ORDER])

    merged = pd.concat(frames, ignore_index=True)
    log.info(
        "Merged %d per-creator CSVs -> %d rows, %d columns",
        len(files), len(merged), len(merged.columns),
    )

    # Sanity: unique segment_ids
    dup = merged["segment_id"].duplicated().sum()
    if dup:
        raise ValueError(f"{dup} duplicate segment_ids in merged output")

    # Report NaN rate in feature columns only (keys/label never null)
    feat_cols = config.FEATURE_COLUMNS
    total_cells = len(merged) * len(feat_cols)
    nan_cells = merged[feat_cols].isna().sum().sum()
    nan_rate = nan_cells / total_cells if total_cells else 0.0
    log.info("Feature NaN rate: %.4f%% (%d / %d cells)", nan_rate * 100, nan_cells, total_cells)
    if nan_rate > 0.005:
        log.warning("NaN rate above 0.5% threshold -- investigate before training")

    # Report label distribution (still pre-relabel)
    pos = int(merged["label"].sum())
    total = len(merged)
    log.info("Pre-relabel label distribution: %d positive / %d total (%.2f%%)",
             pos, total, 100 * pos / total)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.MERGED_FEATURES_CSV, index=False)
    log.info("Wrote %s (%d bytes)",
             config.MERGED_FEATURES_CSV,
             config.MERGED_FEATURES_CSV.stat().st_size)
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    merge()
