"""Post-hoc segment_novelty_to_neighbors pass.

Runs after the pipeline has produced all rows and all raw CLIP embeddings
for a creator's segments. For each segment, computes
  novelty = 1 - mean(cosine_sim(this, prev), cosine_sim(this, next))
where prev/next are the segments at segment_index-1 and +1 in the same video.
First/last segments use only the one available neighbour.
"""
import logging
from typing import Optional

import numpy as np

log = logging.getLogger("clipwhy.features.novelty")


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def apply(
    rows: list[dict],
    clip_embeddings: dict[tuple[str, int], np.ndarray],
) -> list[dict]:
    """Populate each row's `segment_novelty_to_neighbors` field.

    `clip_embeddings` maps (video_id, segment_index) -> raw CLIP embedding
    (pre-PCA, mean-pooled across the 5 sampled frames).
    """
    # Index rows by (video_id, segment_index) for fast neighbour lookup.
    by_key = {(r["video_id"], int(r["segment_index"])): r for r in rows}

    for r in rows:
        vid = r["video_id"]
        idx = int(r["segment_index"])
        this_emb = clip_embeddings.get((vid, idx))
        prev_sim = _cosine(this_emb, clip_embeddings.get((vid, idx - 1)))
        next_sim = _cosine(this_emb, clip_embeddings.get((vid, idx + 1)))
        sims = [s for s in (prev_sim, next_sim) if s is not None]
        if not sims:
            r["segment_novelty_to_neighbors"] = 0.0
        else:
            mean_sim = float(np.mean(sims))
            r["segment_novelty_to_neighbors"] = round(1.0 - mean_sim, 6)
    return rows
