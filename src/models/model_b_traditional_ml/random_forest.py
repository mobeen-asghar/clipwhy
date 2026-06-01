"""Random Forest classifier on the 84 normalised features.

Class imbalance handled with `class_weight='balanced'`. One run per seed
(RANDOM_SEEDS from src.models.shared.seeds). Feature importances saved
per seed for the ablation + explainability layers.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import metrics as eval_metrics
from src.models.shared.data import load_split
from src.models.shared.seeds import RANDOM_SEEDS

N_ESTIMATORS = 100
MAX_DEPTH = None
N_JOBS = -1

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_SUBDIR = RESULTS_DIR / "rf_checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("clipwhy.model_b.rf")


def _fit(seed: int, train_X, train_y):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=seed,
        n_jobs=N_JOBS,
    )
    clf.fit(train_X, train_y)
    return clf


def run(split_eval: str = "test", *, save_models: bool = True) -> dict:
    train = load_split("train")
    test = load_split(split_eval)
    log.info("RF: train=%d (%d pos), %s=%d (%d pos), %d features",
             len(train.y), int(train.y.sum()),
             split_eval, len(test.y), int(test.y.sum()),
             train.X.shape[1])

    per_seed = []
    importances_per_seed = []
    for seed in RANDOM_SEEDS:
        log.info("Fitting RF seed=%d...", seed)
        clf = _fit(seed, train.X, train.y)
        y_score = clf.predict_proba(test.X)[:, 1]
        r = eval_metrics.evaluate(
            test.y, y_score, test.video_ids,
            split=split_eval, model="rf", seed=seed,
        )
        per_seed.append(r)
        importances_per_seed.append(clf.feature_importances_.tolist())
        log.info("  seed=%d: NDCG@10=%.4f AUC=%.4f",
                 seed, r.ndcg_at_k[10], r.auc_roc)
        if save_models:
            mp = MODELS_SUBDIR / f"rf_seed{seed}.pkl"
            with open(mp, "wb") as fh:
                pickle.dump(clf, fh)

    agg = eval_metrics.aggregate_seed_results(per_seed)
    # Mean feature importance across seeds (used by the explainability engine)
    imp = np.asarray(importances_per_seed, dtype=float)
    agg["feature_importance_mean"] = dict(
        zip(train.feature_names, imp.mean(axis=0).tolist())
    )
    agg["per_seed"] = [r.to_dict() for r in per_seed]
    agg["hyperparams"] = {
        "n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
        "class_weight": "balanced", "n_jobs": N_JOBS,
    }

    out = RESULTS_DIR / f"rf_{split_eval}.json"
    out.write_text(json.dumps(agg, indent=2))
    log.info("Wrote %s", out)

    print(f"\n== Random Forest ({split_eval}, 5 seeds) ==")
    for m in ["ndcg_at_10", "precision_at_10", "recall_at_10", "auc_roc", "f1_at_0_5"]:
        md = agg[m]
        print(f"  {m:18s}: mean={md['mean']:.4f} std={md['std']:.4f}")
    top5 = sorted(agg["feature_importance_mean"].items(), key=lambda kv: -kv[1])[:5]
    print("  Top 5 features:")
    for n, v in top5:
        print(f"    {n:35s} {v:.4f}")
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run("test")
