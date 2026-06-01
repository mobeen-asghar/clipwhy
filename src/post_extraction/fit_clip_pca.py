"""Fit PCA(32) on training-split CLIP embeddings, then backfill clip_pca_00..31.

During feature extraction the CLIP PCA columns were written as zeros (see
PLAN.md section 4.6.b and HANDOFF.md section 7 step 3). Raw 768-d CLIP
embeddings were saved to r2:clipwhy-data/clip_embeddings/ per creator so the
PCA fit could run post-extraction. We do that here.

Steps:
1. Load segments_with_splits.csv to know which segment_ids are in train.
2. Load every clip_embeddings/*.npz and concatenate the train rows.
3. Fit PCA(n_components=32, random_state=42) on the train matrix only.
4. Project all 389,619 segments (train/val/test) through the fitted PCA.
5. Backfill the clip_pca_00..31 columns in features_post_pca.csv.
6. Save clip_pca.pkl (the fitted transform) alongside.

Known-missing coverage is handled gracefully:
- CR0195 has no NPZ (video decode failure). Its segments get zero vectors.
- 7 creators with partial CLIP coverage get zero vectors for missing rows.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config

log = logging.getLogger("clipwhy.post_extraction.fit_clip_pca")

PCA_COMPONENTS = config.CLIP_PCA_DIMS  # 32
PCA_SEED = 42


def _load_npz(npz_path) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Load an NPZ produced by the feature extractor.

    File layout (verified 2026-04-21):
      segment_keys: (N,) str, each formatted "{video_id}|{segment_index}"
      embeddings:   (N, 768) float32
    """
    data = np.load(npz_path, allow_pickle=True)
    if "segment_keys" not in data.files or "embeddings" not in data.files:
        raise ValueError(
            f"{npz_path.name}: expected segment_keys + embeddings, got {list(data.files)}"
        )
    raw_keys = data["segment_keys"].tolist()
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    if emb.shape[0] != len(raw_keys):
        raise ValueError(
            f"{npz_path.name}: key/embedding length mismatch ({len(raw_keys)} vs {emb.shape[0]})"
        )

    parsed: list[tuple[str, int]] = []
    for k in raw_keys:
        video_id, _, seg_idx = str(k).rpartition("|")
        if not video_id:
            raise ValueError(f"{npz_path.name}: malformed key {k!r}")
        parsed.append((video_id, int(seg_idx)))
    return parsed, emb


def _build_embedding_table() -> dict[tuple[str, int], np.ndarray]:
    """Concatenates every per-creator NPZ into one {(video_id, seg_idx): vec} dict."""
    npz_files = sorted(config.R2_CLIP_EMBEDDINGS_DIR.glob("CR*_clip_embeddings.npz"))
    log.info("Reading %d CLIP embedding NPZ files...", len(npz_files))

    table: dict[tuple[str, int], np.ndarray] = {}
    emb_dim = None
    for f in npz_files:
        keys, emb = _load_npz(f)
        if emb_dim is None:
            emb_dim = emb.shape[1]
        elif emb.shape[1] != emb_dim:
            raise ValueError(f"{f.name}: dim {emb.shape[1]} != expected {emb_dim}")
        for key, vec in zip(keys, emb):
            table[key] = vec
    log.info("Built embedding table: %d entries, dim=%d", len(table), emb_dim)
    return table


def fit_and_project() -> pd.DataFrame:
    if not config.SEGMENTS_WITH_SPLITS_CSV.exists():
        raise FileNotFoundError(
            f"{config.SEGMENTS_WITH_SPLITS_CSV} not found. Run `split` first."
        )

    df = pd.read_csv(config.SEGMENTS_WITH_SPLITS_CSV)
    total_segments = len(df)
    log.info("Loaded %d segments", total_segments)

    # Build {(video_id, segment_index) -> CLIP vector}
    table = _build_embedding_table()

    keys_in_df = list(zip(df["video_id"].tolist(), df["segment_index"].astype(int).tolist()))
    missing = [k for k in keys_in_df if k not in table]
    log.info("Segments with CLIP vector: %d / %d (missing: %d)",
             total_segments - len(missing), total_segments, len(missing))

    emb_dim = 768
    # Assemble full matrix in segment-row order, zero for missing (CR0195 etc)
    full = np.zeros((total_segments, emb_dim), dtype=np.float32)
    for i, key in enumerate(keys_in_df):
        v = table.get(key)
        if v is not None:
            full[i] = v

    # Fit on train rows with non-zero embedding
    is_train = (df["split"] == "train").to_numpy()
    has_emb = np.array([k in table for k in keys_in_df])
    fit_mask = is_train & has_emb
    X_fit = full[fit_mask]
    log.info("PCA fit matrix: %d rows x %d dims (train-only, non-missing)", *X_fit.shape)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=PCA_SEED)
    pca.fit(X_fit)
    total_var = float(pca.explained_variance_ratio_.sum())
    log.info("PCA fit complete. Explained variance (cum, first %d comps): %.3f",
             PCA_COMPONENTS, total_var)

    # Project every row (train/val/test), keep zeros at missing rows as-is.
    # Transforming a zero vector through PCA gives the projection of the origin
    # offset by the mean centering; we explicitly zero them out after for clarity.
    projected = pca.transform(full)
    projected[~has_emb] = 0.0  # explicit zero row for missing embeddings

    # Write back into the DataFrame
    for i, col in enumerate(config.CLIP_PCA_COLUMNS):
        df[col] = projected[:, i].astype(np.float32)

    # Persist outputs
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CLIP_PCA_PKL, "wb") as fh:
        pickle.dump({"pca": pca, "n_components": PCA_COMPONENTS, "seed": PCA_SEED,
                     "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                     "missing_segment_ids_count": len(missing)}, fh)
    df.to_csv(config.FEATURES_POST_PCA_CSV, index=False)
    log.info("Wrote %s and %s", config.FEATURES_POST_PCA_CSV, config.CLIP_PCA_PKL)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit_and_project()
