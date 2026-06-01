"""Rule-based baseline.

Hand-picked weighted sum of 7 features, no training. Lower bound for RQ2:
if ML can't beat hand-coded rules, the ML is adding nothing. Adapted from
V1 (clipwhy-pipeline/src/models/model_a_baselines/rule_based.py) with
VADER sentiment features replaced by audio emotion equivalents.

V1 weights   -> V2 weights (same structure, features swapped)
-----------------------------------------------
hook_word_ratio          0.20  ->  hook_word_ratio          0.20
energy_mean              0.20  ->  energy_mean              0.20
emotional_intensity      0.15  ->  arousal_mean             0.15
words_per_second         0.15  ->  words_per_second         0.15
sentiment_arc_range      0.10  ->  arousal_arc_direction    0.10
is_intro                 0.10  ->  is_intro                 0.10
speaking_rate_audio      0.10  ->  speaking_rate_audio      0.10

Features are consumed from feature_matrix_test.csv (already z-score
normalised) and passed through a sigmoid so the score is in [0, 1]. We
still rank segments by the raw weighted sum; the sigmoid is only a
cosmetic bound. The rule-based model is deterministic so a single
evaluation is sufficient (no seed loop).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation import metrics as eval_metrics
from src.models.shared.data import load_split

WEIGHTS = {
    "hook_word_ratio": 0.20,
    "energy_mean": 0.20,
    "arousal_mean": 0.15,
    "words_per_second": 0.15,
    "arousal_arc_direction": 0.10,
    "is_intro": 0.10,
    "speaking_rate_audio": 0.10,
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("clipwhy.model_a.rule_based")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score(data) -> np.ndarray:
    feat_idx = {name: i for i, name in enumerate(data.feature_names)}
    missing = [f for f in WEIGHTS if f not in feat_idx]
    if missing:
        raise ValueError(f"Rule-based: features missing: {missing}")
    weighted_sum = np.zeros(len(data.y), dtype=np.float64)
    for feat, w in WEIGHTS.items():
        weighted_sum += w * data.X[:, feat_idx[feat]].astype(np.float64)
    return _sigmoid(weighted_sum).astype(np.float32)


def run(split: str = "test") -> dict:
    data = load_split(split)
    log.info("Rule-based: %s split, %d segments, %d positives",
             split, len(data.y), int(data.y.sum()))
    y_score = score(data)
    res = eval_metrics.evaluate(
        data.y, y_score, data.video_ids,
        split=split, model="rule_based",
        notes=f"weights={WEIGHTS}",
    )
    out_dict = res.to_dict()
    out_dict["weights"] = WEIGHTS

    out = RESULTS_DIR / f"rule_based_{split}.json"
    out.write_text(json.dumps(out_dict, indent=2))
    log.info("Wrote %s", out)

    print(f"\n== Rule-based ({split}) ==")
    print(f"  NDCG@10       {res.ndcg_at_k[10]:.4f}")
    print(f"  Precision@10  {res.precision_at_k[10]:.4f}")
    print(f"  Recall@10     {res.recall_at_k[10]:.4f}")
    print(f"  AUC-ROC       {res.auc_roc:.4f}")
    print(f"  F1@0.5        {res.f1_at_0_5:.4f}")
    print(f"  videos scored {res.n_rank_eligible_videos}/{res.n_videos}")
    return out_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run("test")
