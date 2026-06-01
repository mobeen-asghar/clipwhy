"""Paths and constants for the post-extraction pipeline.

The R2 bucket is read-only from this module's perspective. Everything is pulled
to data/r2_cache/ once, then all computation happens on local files.
Outputs are written to data/post_extraction/ and not pushed anywhere.
"""
from pathlib import Path

from src.feature_extraction.config import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    META_COLUMNS,
    OUTPUT_COLUMN_ORDER,
    CATEGORY_ORDER,
    CLIP_PCA_DIMS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# R2 local mirror (read-only view of r2:clipwhy-data/)
R2_CACHE = DATA_ROOT / "r2_cache"
R2_FEATURES_DIR = R2_CACHE / "features"
R2_CLIP_EMBEDDINGS_DIR = R2_CACHE / "clip_embeddings"
R2_LABELED_DIR = R2_CACHE / "labeled"

# Outputs of this pipeline
OUT_DIR = DATA_ROOT / "post_extraction"
MERGED_FEATURES_CSV = OUT_DIR / "features.csv"
SEGMENTS_WITH_SPLITS_CSV = OUT_DIR / "segments_with_splits.csv"
CLIP_PCA_PKL = OUT_DIR / "clip_pca.pkl"
FEATURES_POST_PCA_CSV = OUT_DIR / "features_post_pca.csv"
NORMALISATION_PARAMS_JSON = OUT_DIR / "normalisation_params.json"
FEATURE_MATRIX_TRAIN_RAW = OUT_DIR / "feature_matrix_train_raw.csv"
FEATURE_MATRIX_TRAIN = OUT_DIR / "feature_matrix_train.csv"
FEATURE_MATRIX_VAL = OUT_DIR / "feature_matrix_val.csv"
FEATURE_MATRIX_TEST = OUT_DIR / "feature_matrix_test.csv"

# R2 prefixes (for rclone commands, read-only)
R2_REMOTE = "r2:clipwhy-data"
R2_FEATURES_PREFIX = f"{R2_REMOTE}/features/"
R2_CLIP_EMBEDDINGS_PREFIX = f"{R2_REMOTE}/clip_embeddings/"
R2_LABELED_PREFIX = f"{R2_REMOTE}/labeled/"

# Split config (identical to V1 clipwhy-pipeline/src/split_data.py)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_RANDOM_STATE = 42

# Feature columns to clip to [0, 1] before normalisation
# (handles <0.2% overshoot documented in HANDOFF.md section 7)
CLIP_TO_UNIT_INTERVAL = [
    "arousal_mean", "valence_mean", "dominance_mean",
    "arousal_peak", "music_presence",
]

# Binary columns: no z-score, keep 0/1 as-is
BINARY_COLUMNS = [
    "is_intro", "is_outro", "is_first_segment", "is_last_segment",
    *[f"creator_category_{c}" for c in CATEGORY_ORDER],
]

# CLIP PCA columns (filled by fit_clip_pca.py, zero in per-creator CSVs during extraction)
CLIP_PCA_COLUMNS = [f"clip_pca_{i:02d}" for i in range(CLIP_PCA_DIMS)]


def ensure_dirs() -> None:
    for d in [
        R2_CACHE, R2_FEATURES_DIR, R2_CLIP_EMBEDDINGS_DIR, R2_LABELED_DIR, OUT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
