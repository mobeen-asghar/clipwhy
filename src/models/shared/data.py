"""Shared data loading for every model.

All models read from the same feature_matrix_{train,val,test}.csv files
produced by src.post_extraction. This helper returns numpy arrays ready
for scikit-learn / xgboost and also gives back the metadata columns
needed for per-video ranking metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.post_extraction import config as post_config


class SplitData(NamedTuple):
    X: np.ndarray          # (n_segments, n_features) float32
    y: np.ndarray          # (n_segments,) int {0, 1}
    video_ids: np.ndarray  # (n_segments,) str
    segment_ids: np.ndarray
    categories: np.ndarray
    feature_names: list[str]


def load_split(split: str, *, feature_cols: list[str] | None = None) -> SplitData:
    """Load one split (train|val|test) from the normalised feature matrix."""
    path_map = {
        "train": post_config.FEATURE_MATRIX_TRAIN,
        "val": post_config.FEATURE_MATRIX_VAL,
        "test": post_config.FEATURE_MATRIX_TEST,
    }
    if split not in path_map:
        raise ValueError(f"split must be train|val|test, got {split}")
    df = pd.read_csv(path_map[split])

    feature_cols = feature_cols or list(post_config.FEATURE_COLUMNS)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {split}: {missing[:5]}")

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].to_numpy(dtype=np.int32, copy=True)
    return SplitData(
        X=X,
        y=y,
        video_ids=df["video_id"].to_numpy(copy=True),
        segment_ids=df["segment_id"].to_numpy(copy=True),
        categories=df["category"].to_numpy(copy=True),
        feature_names=list(feature_cols),
    )


def load_raw_train() -> pd.DataFrame:
    """Un-normalised training set (used by the explainability layer for
    human-readable feature values)."""
    return pd.read_csv(post_config.FEATURE_MATRIX_TRAIN_RAW)


def feature_categories() -> dict[str, list[str]]:
    """Feature grouping by category, for ablation studies.

    Grouped to match the 8 categories used in V2 PLAN.md section 4.
    """
    groups = {
        "text": [
            "word_count", "words_per_second", "hook_word_count", "hook_word_ratio",
            "question_count", "question_density", "second_person_ratio",
            "first_person_ratio", "first_5s_hook_word_ratio", "articulation_rate",
        ],
        "audio_speech": [
            "energy_mean", "energy_var", "energy_first_3s_ratio",
            "pitch_range", "pitch_std", "speaking_rate_audio", "silence_ratio",
        ],
        "voice_quality": ["jitter_local", "shimmer_local"],
        "audio_events": [
            "music_presence", "music_fraction", "speech_music_ratio", "laughter_peak",
        ],
        "audio_emotion": [
            "arousal_mean", "valence_mean", "dominance_mean",
            "arousal_std", "arousal_peak",
            "arousal_arc_direction", "valence_arc_direction",
        ],
        "visual": (
            ["dover_aesthetic_score", "dover_technical_score"]
            + [f"clip_pca_{i:02d}" for i in range(32)]
            + ["colorfulness", "brightness_mean"]
            + ["cut_count", "cuts_per_second"]
            + ["face_present_ratio", "largest_face_area_ratio_max", "face_count_median"]
        ),
        "structural": [
            "position_ratio", "is_intro", "is_outro",
            "segment_duration", "video_duration",
            "is_first_segment", "is_last_segment",
            "segment_novelty_to_neighbors",
        ],
        "creator_context": [f"creator_category_{c}" for c in post_config.CATEGORY_ORDER],
    }
    # Sanity: groups should partition FEATURE_COLUMNS
    all_in_groups = [f for fs in groups.values() for f in fs]
    expected = set(post_config.FEATURE_COLUMNS)
    if set(all_in_groups) != expected:
        missing = expected - set(all_in_groups)
        extra = set(all_in_groups) - expected
        raise ValueError(f"category groups mismatch. missing={missing} extra={extra}")
    return groups
