"""Clip emotion features to [0,1], then z-score normalise from train stats only.

Uses the same approach V1 took in clipwhy-pipeline/src/feature_extraction/pipeline.py:
- Stats (mean, std) are computed from the TRAINING split only.
- The same mean/std are applied to train, val, and test.
- Binary features (is_intro, is_outro, is_first_segment, is_last_segment,
  creator_category_*) are passed through unchanged so they remain 0/1.

Pre-step: columns listed in CLIP_TO_UNIT_INTERVAL (audio emotion + music_presence)
are clipped to [0, 1] to handle the <0.2% overshoots documented in
HANDOFF.md section 7. This is done before computing train stats so the
clipped values drive normalisation.

Outputs:
- data/post_extraction/feature_matrix_train_raw.csv  (un-normalised train)
- data/post_extraction/feature_matrix_train.csv      (z-score normalised)
- data/post_extraction/feature_matrix_val.csv        (z-score using train params)
- data/post_extraction/feature_matrix_test.csv       (z-score using train params)
- data/post_extraction/normalisation_params.json     (mean/std per feature)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("clipwhy.post_extraction.normalise")


def _clip_unit_interval(df: pd.DataFrame) -> pd.DataFrame:
    for col in config.CLIP_TO_UNIT_INTERVAL:
        if col not in df.columns:
            log.warning("Missing column to clip: %s", col)
            continue
        before_over = int((df[col] > 1.0).sum())
        before_under = int((df[col] < 0.0).sum())
        df[col] = df[col].clip(lower=0.0, upper=1.0)
        if before_over or before_under:
            log.info("Clipped %s to [0,1]: %d > 1.0 and %d < 0.0",
                     col, before_over, before_under)
    return df


def _write_matrix(df: pd.DataFrame, path: Path, *, feature_cols: list[str]) -> None:
    keep_cols = ["segment_id", "video_id", "creator_id", "category",
                 "segment_index", "label", "split"] + feature_cols
    subset = df[[c for c in keep_cols if c in df.columns]]
    subset.to_csv(path, index=False)
    log.info("Wrote %s (%d rows, %d cols)", path, len(subset), len(subset.columns))


def normalise() -> None:
    if not config.FEATURES_POST_PCA_CSV.exists():
        raise FileNotFoundError(
            f"{config.FEATURES_POST_PCA_CSV} not found. Run `pca` first."
        )

    df = pd.read_csv(config.FEATURES_POST_PCA_CSV)
    log.info("Loaded %d rows from %s", len(df), config.FEATURES_POST_PCA_CSV.name)

    df = _clip_unit_interval(df)

    feature_cols = list(config.FEATURE_COLUMNS)
    binary_cols = set(config.BINARY_COLUMNS)
    numeric_cols = [c for c in feature_cols if c not in binary_cols]

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    log.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    # Write un-normalised train first (for inspection / explainability layer)
    _write_matrix(train, config.FEATURE_MATRIX_TRAIN_RAW, feature_cols=feature_cols)

    # Compute train stats on numeric features only
    means = train[numeric_cols].mean()
    stds = train[numeric_cols].std(ddof=0)
    # Guard against zero std (constant features): replace std=0 with 1 so x - mean stays 0
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        log.warning("Zero-std columns in train (will be set to 0 after norm): %s", zero_std_cols)
    stds_safe = stds.replace(0, 1.0)

    params = {
        col: {"mean": float(means[col]), "std": float(stds[col])}
        for col in numeric_cols
    }
    with open(config.NORMALISATION_PARAMS_JSON, "w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    log.info("Wrote %s (%d features)", config.NORMALISATION_PARAMS_JSON, len(params))

    # Apply z-score to train/val/test
    for split_df in (train, val, test):
        split_df[numeric_cols] = (
            (split_df[numeric_cols].to_numpy() - means.to_numpy()) / stds_safe.to_numpy()
        )
        for col in zero_std_cols:
            split_df[col] = 0.0

    _write_matrix(train, config.FEATURE_MATRIX_TRAIN, feature_cols=feature_cols)
    _write_matrix(val, config.FEATURE_MATRIX_VAL, feature_cols=feature_cols)
    _write_matrix(test, config.FEATURE_MATRIX_TEST, feature_cols=feature_cols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    normalise()
