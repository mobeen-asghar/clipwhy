"""Train/val/test split by video_id (not segment).

Ports the approach from V1 (clipwhy-pipeline/src/split_data.py). Using
GroupShuffleSplit with random_state=42 and 70/15/15 ratios, grouped on
video_id, guarantees no long-form video appears in more than one split.
This prevents leakage where a model could memorise segments from the same
video across train/test.

Writes data/post_extraction/segments_with_splits.csv: the merged feature
matrix with an added `split` column, ready for downstream PCA, normalise,
and training.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config

log = logging.getLogger("clipwhy.post_extraction.split")


def split(features_csv=None) -> pd.DataFrame:
    features_csv = features_csv or config.MERGED_FEATURES_CSV
    if not features_csv.exists():
        raise FileNotFoundError(f"{features_csv} not found. Run `merge` first.")

    df = pd.read_csv(features_csv)
    log.info("Loaded %d segments across %d videos and %d creators",
             len(df), df["video_id"].nunique(), df["creator_id"].nunique())

    # Positive count before split
    total_pos = int(df["label"].sum())
    log.info("Total positive segments: %d (%.2f%%)", total_pos, 100 * total_pos / len(df))

    seed = config.SPLIT_RANDOM_STATE
    # First pass: 70% train vs 30% holdout
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(gss_1.split(df, groups=df["video_id"]))

    # Second pass: split holdout 50/50 into val and test (15% each of original)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(gss_2.split(holdout_df, groups=holdout_df["video_id"]))
    val_idx = holdout_idx[val_local]
    test_idx = holdout_idx[test_local]

    split_col = np.empty(len(df), dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"
    df["split"] = split_col

    # Integrity: every video in exactly one split
    split_per_video = df.groupby("video_id")["split"].nunique()
    bad = split_per_video[split_per_video > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} videos appear in multiple splits: {bad.index.tolist()[:5]}")

    # Report counts
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        log.info(
            "split=%-5s: %d segments, %d videos, %d positives (%.2f%%)",
            sp, len(sub), sub["video_id"].nunique(),
            int(sub["label"].sum()),
            100 * sub["label"].sum() / max(len(sub), 1),
        )

    # Per-category per-split positive breakdown (sanity for stratification)
    pivot = (
        df.assign(label=df["label"].astype(int))
          .pivot_table(index="category", columns="split", values="label",
                       aggfunc="sum", fill_value=0)
    )
    log.info("Per-category positives by split:\n%s", pivot.to_string())

    df.to_csv(config.SEGMENTS_WITH_SPLITS_CSV, index=False)
    log.info("Wrote %s", config.SEGMENTS_WITH_SPLITS_CSV)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split()
