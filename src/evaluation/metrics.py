"""Evaluation metrics for ClipWhy V2.

Mirrors clipwhy-pipeline/src/evaluation/metrics.py. Primary metrics are
ranking metrics computed PER VIDEO (ranking segments within a single
video), then averaged across videos. Videos without any positive segment
are excluded from the ranking average because there is nothing to rank.

Also exposes classification metrics (AUC-ROC, F1@0.5) computed globally
across the whole split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, ndcg_score, roc_auc_score

K_VALUES = (3, 5, 10)


@dataclass
class EvalResult:
    split: str
    model: str
    n_segments: int
    n_videos: int
    n_positives: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    auc_roc: float
    f1_at_0_5: float
    # Videos with >= 1 positive (used in ranking averages)
    n_rank_eligible_videos: int
    per_video_metrics: list[dict] = field(default_factory=list)
    seed: int | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "model": self.model,
            "seed": self.seed,
            "n_segments": self.n_segments,
            "n_videos": self.n_videos,
            "n_positives": self.n_positives,
            "n_rank_eligible_videos": self.n_rank_eligible_videos,
            "precision_at_k": {str(k): float(v) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): float(v) for k, v in self.recall_at_k.items()},
            "ndcg_at_k": {str(k): float(v) for k, v in self.ndcg_at_k.items()},
            "auc_roc": float(self.auc_roc),
            "f1_at_0_5": float(self.f1_at_0_5),
            "notes": self.notes,
        }


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    k = min(k, len(y_true))
    order = np.argsort(-y_score)
    top_k = y_true[order[:k]]
    n_pos = int(y_true.sum())
    precision = float(top_k.sum()) / k
    recall = float(top_k.sum()) / n_pos if n_pos > 0 else 0.0
    return precision, recall


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) < 2:
        return 0.0
    k = min(k, len(y_true))
    return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    video_ids: np.ndarray,
    *,
    split: str,
    model: str,
    seed: int | None = None,
    k_values: Iterable[int] = K_VALUES,
    notes: str = "",
) -> EvalResult:
    """Compute ranking metrics per-video (averaged) + classification metrics globally.

    Args:
        y_true: (N,) binary labels.
        y_score: (N,) predicted probability or score (higher = more viral).
        video_ids: (N,) string group identifier. Ranking is within each group.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    video_ids = np.asarray(video_ids)
    assert y_true.shape == y_score.shape == video_ids.shape

    per_video_metrics: list[dict] = []
    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    unique_videos = np.unique(video_ids)

    n_rank_eligible_videos = 0
    for vid in unique_videos:
        mask = video_ids == vid
        vt = y_true[mask]
        vs = y_score[mask]
        if vt.sum() == 0:
            # Can't rank what doesn't exist -- skip from ranking averages
            continue
        n_rank_eligible_videos += 1
        vid_row = {"video_id": str(vid), "n_segments": int(mask.sum()), "n_positives": int(vt.sum())}
        for k in k_values:
            p, r = _precision_recall_at_k(vt, vs, k)
            n = _ndcg_at_k(vt, vs, k)
            precisions[k].append(p)
            recalls[k].append(r)
            ndcgs[k].append(n)
            vid_row[f"precision_at_{k}"] = p
            vid_row[f"recall_at_{k}"] = r
            vid_row[f"ndcg_at_{k}"] = n
        per_video_metrics.append(vid_row)

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    precision_at_k = {k: _avg(precisions[k]) for k in k_values}
    recall_at_k = {k: _avg(recalls[k]) for k in k_values}
    ndcg_at_k = {k: _avg(ndcgs[k]) for k in k_values}

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    y_pred = (y_score >= 0.5).astype(int)
    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        f1 = 0.0

    return EvalResult(
        split=split,
        model=model,
        n_segments=int(y_true.size),
        n_videos=int(len(unique_videos)),
        n_positives=int(y_true.sum()),
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        auc_roc=auc,
        f1_at_0_5=f1,
        n_rank_eligible_videos=n_rank_eligible_videos,
        per_video_metrics=per_video_metrics,
        seed=seed,
        notes=notes,
    )


def aggregate_seed_results(results: list[EvalResult]) -> dict:
    """Aggregate N single-seed results into {mean, std} per metric."""
    if not results:
        return {}
    metric_arrays: dict[str, list[float]] = {}
    for r in results:
        for k, v in r.precision_at_k.items():
            metric_arrays.setdefault(f"precision_at_{k}", []).append(v)
        for k, v in r.recall_at_k.items():
            metric_arrays.setdefault(f"recall_at_{k}", []).append(v)
        for k, v in r.ndcg_at_k.items():
            metric_arrays.setdefault(f"ndcg_at_{k}", []).append(v)
        metric_arrays.setdefault("auc_roc", []).append(r.auc_roc)
        metric_arrays.setdefault("f1_at_0_5", []).append(r.f1_at_0_5)

    agg = {}
    for name, vals in metric_arrays.items():
        arr = np.asarray(vals, dtype=float)
        agg[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_seeds": len(arr),
            "values": arr.tolist(),
        }
    agg["seeds"] = [r.seed for r in results]
    agg["model"] = results[0].model
    agg["split"] = results[0].split
    return agg
