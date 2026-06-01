"""Per-category feature ablation study (answers RQ1).

Mirrors V1 protocol from clipwhy-pipeline/src/evaluation/ablation_study.py.
Two analyses, both on XGBoost (the strongest tabular model in V2):

1. REMOVE-ONE-CATEGORY: train XGBoost with all 84 features minus one
   category, see how much AUC drops vs the full-feature baseline.
   Bigger drop = more important category.

2. SINGLE-CATEGORY-ISOLATION: train XGBoost with ONLY that category's
   features. Tells us how much signal lives in each category alone,
   independent of interactions with other categories.

Each ablation runs 5 seeds for statistical comparability with the
existing model results. Total: 8 categories x 2 modes x 5 seeds = 80
XGBoost fits, ~10 min on any modern CPU.

Outputs:
  data/post_extraction/results/ablation_xgboost.json
  data/post_extraction/results/ABLATION_FINDINGS.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.evaluation import metrics as eval_metrics
from src.evaluation.statistical_tests import compare
from src.models.shared.data import feature_categories, load_split
from src.models.shared.seeds import RANDOM_SEEDS

# XGBoost hyperparameters (same as model_b_traditional_ml/xgboost_model.py)
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
EARLY_STOP = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "post_extraction" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_JSON = RESULTS_DIR / "ablation_xgboost.json"
ABLATION_MD = RESULTS_DIR / "ABLATION_FINDINGS.md"

log = logging.getLogger("clipwhy.ablation")


def _train_one_seed(seed, X_train, y_train, X_val, y_val):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
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
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def _slice(X, all_features, keep_features):
    keep_idx = [i for i, n in enumerate(all_features) if n in keep_features]
    return X[:, keep_idx]


def run_ablation(label, keep_features, train, val, test):
    """Train + evaluate 5 seeds with the given subset of features."""
    X_train = _slice(train.X, train.feature_names, keep_features)
    X_val = _slice(val.X, val.feature_names, keep_features)
    X_test = _slice(test.X, test.feature_names, keep_features)
    log.info("[%s] features=%d, train=(%d, %d), test=(%d, %d)",
             label, X_train.shape[1], *X_train.shape, *X_test.shape)

    seed_results = []
    for seed in RANDOM_SEEDS:
        clf = _train_one_seed(seed, X_train, train.y, X_val, val.y)
        y_score = clf.predict_proba(X_test)[:, 1]
        r = eval_metrics.evaluate(
            test.y, y_score, test.video_ids,
            split="test", model=label, seed=seed,
        )
        seed_results.append(r)
        log.info("  seed=%d: AUC=%.4f NDCG@10=%.4f",
                 seed, r.auc_roc, r.ndcg_at_k[10])

    agg = eval_metrics.aggregate_seed_results(seed_results)
    agg["features_count"] = X_train.shape[1]
    agg["features_kept"] = sorted(keep_features)
    return agg, seed_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    log.info("Loaded splits: train=%d val=%d test=%d (test positives=%d)",
             len(train.y), len(val.y), len(test.y), int(test.y.sum()))

    groups = feature_categories()
    all_features = set(train.feature_names)
    log.info("Feature categories (8 total):")
    for cat, feats in groups.items():
        log.info("  %-18s %d features", cat, len(feats))

    # 0. Baseline: full feature set (mirrors xgboost_test.json)
    log.info("\n=== BASELINE: full 84 features ===")
    baseline_agg, baseline_seeds = run_ablation(
        "xgboost_full", all_features, train, val, test,
    )
    baseline_aucs = [r.auc_roc for r in baseline_seeds]

    # 1. Remove-one-category
    remove_one = {}
    for cat, feats in groups.items():
        log.info("\n=== REMOVE-ONE: %s (drops %d features) ===", cat, len(feats))
        keep = all_features - set(feats)
        agg, seeds = run_ablation(f"remove_{cat}", keep, train, val, test)
        ablated_aucs = [r.auc_roc for r in seeds]
        cmp = compare(baseline_aucs, ablated_aucs,
                      label_a="full", label_b=f"-{cat}")
        agg["ttest_vs_full"] = cmp
        agg["delta_auc_vs_full"] = float(np.mean(ablated_aucs) - np.mean(baseline_aucs))
        remove_one[cat] = agg

    # 2. Single-category-isolation
    isolation = {}
    for cat, feats in groups.items():
        log.info("\n=== ISOLATE: %s only (%d features) ===", cat, len(feats))
        keep = set(feats)
        agg, seeds = run_ablation(f"only_{cat}", keep, train, val, test)
        agg["delta_auc_vs_full"] = float(np.mean([r.auc_roc for r in seeds]) - np.mean(baseline_aucs))
        isolation[cat] = agg

    # Save the full result
    out = {
        "baseline_xgboost_full": baseline_agg,
        "remove_one_category": remove_one,
        "single_category_isolation": isolation,
        "categories": {k: list(v) for k, v in groups.items()},
        "hyperparams": {
            "model": "xgboost",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "early_stopping_rounds": EARLY_STOP,
            "seeds": list(RANDOM_SEEDS),
        },
    }
    ABLATION_JSON.write_text(json.dumps(out, indent=2))
    log.info("\nWrote %s", ABLATION_JSON)

    # Markdown report
    lines = ["# Per-Category Ablation Study (XGBoost, V2)\n"]
    lines.append(f"Baseline XGBoost AUC: **{baseline_agg['auc_roc']['mean']:.4f} +/- {baseline_agg['auc_roc']['std']:.4f}** (84 features, 5 seeds)\n")

    lines.append("## Remove-One-Category (importance = how much AUC drops)\n")
    lines.append("| Category | Features removed | AUC after | Δ AUC | t-stat | p-value | Cohen's d | Significant? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    sorted_remove = sorted(remove_one.items(),
                            key=lambda kv: kv[1]["delta_auc_vs_full"])
    for cat, agg in sorted_remove:
        c = agg["ttest_vs_full"]
        sig = "YES" if c["significant_at_alpha"] else "no"
        lines.append(
            f"| {cat} | {agg['features_count']} kept ({84-agg['features_count']} removed) "
            f"| {agg['auc_roc']['mean']:.4f} +/- {agg['auc_roc']['std']:.4f} "
            f"| {agg['delta_auc_vs_full']:+.4f} "
            f"| {c['t_stat']:+.2f} | {c['p_value']:.4f} | {c['cohens_d']:+.2f} | {sig} |"
        )

    lines.append("\n## Single-Category-Isolation (only that category's features)\n")
    lines.append("| Category | Features used | AUC | Δ vs full | Above random (0.50)? |")
    lines.append("|---|---|---|---|---|")
    sorted_iso = sorted(isolation.items(),
                        key=lambda kv: -kv[1]["auc_roc"]["mean"])
    for cat, agg in sorted_iso:
        delta = agg["delta_auc_vs_full"]
        above_rand = "YES" if agg["auc_roc"]["mean"] > 0.55 else "barely" if agg["auc_roc"]["mean"] > 0.50 else "NO"
        lines.append(
            f"| {cat} | {agg['features_count']} "
            f"| {agg['auc_roc']['mean']:.4f} +/- {agg['auc_roc']['std']:.4f} "
            f"| {delta:+.4f} | {above_rand} |"
        )

    ABLATION_MD.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", ABLATION_MD)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
