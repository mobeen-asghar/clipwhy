"""XGBoost classifier on the 84 normalised features.

Class imbalance handled with `scale_pos_weight` = n_neg / n_pos. Early
stopping on val AUC-PR. One run per seed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.evaluation import metrics as eval_metrics
from src.models.shared.data import load_split
from src.models.shared.seeds import RANDOM_SEEDS

N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
EARLY_STOP = 20

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_SUBDIR = RESULTS_DIR / "xgb_checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("clipwhy.model_b.xgb")


def _fit(seed, train, val):
    n_neg = int((train.y == 0).sum())
    n_pos = int((train.y == 1).sum())
    spw = n_neg / max(n_pos, 1)
    clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=spw,
        random_state=seed,
        eval_metric="aucpr",
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOP,
        tree_method="hist",
    )
    clf.fit(train.X, train.y, eval_set=[(val.X, val.y)], verbose=False)
    return clf, spw


def run(split_eval: str = "test", *, save_models: bool = True) -> dict:
    train = load_split("train")
    val = load_split("val")
    test_data = load_split(split_eval)
    log.info("XGBoost: train=%d (%d pos), val=%d, %s=%d, %d features",
             len(train.y), int(train.y.sum()), len(val.y),
             split_eval, len(test_data.y), train.X.shape[1])

    per_seed = []
    importances_per_seed = []
    spw_value = None
    for seed in RANDOM_SEEDS:
        log.info("Fitting XGBoost seed=%d...", seed)
        clf, spw_value = _fit(seed, train, val)
        y_score = clf.predict_proba(test_data.X)[:, 1]
        r = eval_metrics.evaluate(
            test_data.y, y_score, test_data.video_ids,
            split=split_eval, model="xgboost", seed=seed,
        )
        per_seed.append(r)
        importances_per_seed.append(clf.feature_importances_.tolist())
        best_iter = getattr(clf, "best_iteration", None)
        log.info("  seed=%d: NDCG@10=%.4f AUC=%.4f best_iter=%s",
                 seed, r.ndcg_at_k[10], r.auc_roc, best_iter)
        if save_models:
            mp = MODELS_SUBDIR / f"xgb_seed{seed}.json"
            clf.save_model(mp)

    agg = eval_metrics.aggregate_seed_results(per_seed)
    imp = np.asarray(importances_per_seed, dtype=float)
    agg["feature_importance_mean"] = dict(
        zip(train.feature_names, imp.mean(axis=0).tolist())
    )
    agg["per_seed"] = [r.to_dict() for r in per_seed]
    agg["hyperparams"] = {
        "n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "scale_pos_weight": float(spw_value),
        "early_stopping_rounds": EARLY_STOP,
        "tree_method": "hist", "eval_metric": "aucpr",
    }
    out = RESULTS_DIR / f"xgboost_{split_eval}.json"
    out.write_text(json.dumps(agg, indent=2))
    log.info("Wrote %s", out)

    print(f"\n== XGBoost ({split_eval}, 5 seeds, scale_pos_weight={spw_value:.1f}) ==")
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
