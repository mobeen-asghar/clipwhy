"""Aggregate all model results and run paired t-tests for the FPR.

Reads results/*_test.json from every model folder, produces a single
summary JSON plus a Markdown report at data/post_extraction/results/
MODEL_COMPARISON.md. Only models whose results files exist will appear;
missing models (e.g. Model C / D pending a GPU run) are noted.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.statistical_tests import compare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "src" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "post_extraction" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILES = {
    "random":                   MODELS_ROOT / "model_a_baselines" / "results" / "random_baseline_test.json",
    "rule_based":               MODELS_ROOT / "model_a_baselines" / "results" / "rule_based_test.json",
    "rf":                       MODELS_ROOT / "model_b_traditional_ml" / "results" / "rf_test.json",
    "xgboost":                  MODELS_ROOT / "model_b_traditional_ml" / "results" / "xgboost_test.json",
    "bert":                     MODELS_ROOT / "model_c_bert" / "results" / "bert_test.json",
    # 3-branch multimodal (V1-style). Old code wrote multimodal_test.json;
    # new code writes multimodal_original_test.json. Pick whichever exists.
    "multimodal":               MODELS_ROOT / "model_d_multimodal" / "results" / "multimodal_test.json",
    "multimodal_original":      MODELS_ROOT / "model_d_multimodal" / "results" / "multimodal_original_test.json",
    # 4-branch multimodal with metadata -- written by the standalone pod
    "multimodal_with_metadata": MODELS_ROOT / "model_d_multimodal" / "results" / "multimodal_with_metadata_test.json",
}

KEY_METRICS = ["ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
               "precision_at_10", "recall_at_10",
               "auc_roc", "f1_at_0_5"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _summary_row(name: str, data: dict) -> dict:
    """Handles both single-run (rule_based) and multi-seed aggregates."""
    row = {"model": name, "n_seeds": 0}
    if data is None:
        return row | {"status": "missing"}
    # Multi-seed result: has keys like 'ndcg_at_10' -> {'mean', 'std'}
    if "auc_roc" in data and isinstance(data["auc_roc"], dict):
        row["n_seeds"] = data["auc_roc"].get("n_seeds", 0)
        for m in KEY_METRICS:
            if m in data:
                row[f"{m}_mean"] = data[m]["mean"]
                row[f"{m}_std"] = data[m]["std"]
        return row
    # Single-run result (EvalResult.to_dict())
    row["n_seeds"] = 1
    for m in KEY_METRICS:
        if m.startswith("ndcg_at_"):
            k = m.split("_")[-1]
            row[f"{m}_mean"] = data["ndcg_at_k"][k]
        elif m.startswith("precision_at_"):
            k = m.split("_")[-1]
            row[f"{m}_mean"] = data["precision_at_k"][k]
        elif m.startswith("recall_at_"):
            k = m.split("_")[-1]
            row[f"{m}_mean"] = data["recall_at_k"][k]
        else:
            row[f"{m}_mean"] = data.get(m, None)
        row[f"{m}_std"] = 0.0
    return row


def _per_seed_scores(data: dict | None, metric: str) -> list[float] | None:
    """Extract raw per-seed values for paired tests."""
    if data is None:
        return None
    if metric in data and isinstance(data[metric], dict) and "values" in data[metric]:
        return data[metric]["values"]
    return None


def main() -> None:
    loaded = {name: _load(p) for name, p in RESULT_FILES.items()}
    rows = [_summary_row(name, data) for name, data in loaded.items()]

    # Paired tests on AUC-ROC (available where both models have 5+ seeds)
    comparisons = []
    model_names = [n for n, d in loaded.items() if d is not None]
    for i, a in enumerate(model_names):
        for b in model_names[i + 1:]:
            a_scores = _per_seed_scores(loaded[a], "auc_roc")
            b_scores = _per_seed_scores(loaded[b], "auc_roc")
            if a_scores and b_scores and len(a_scores) == len(b_scores) and len(a_scores) >= 2:
                comparisons.append(compare(a_scores, b_scores, label_a=a, label_b=b))

    summary = {"per_model": rows, "pairwise_ttests_auc_roc": comparisons}
    (OUT_DIR / "model_comparison.json").write_text(json.dumps(summary, indent=2))

    # Markdown
    lines = ["# Model Comparison (test split)\n"]
    lines.append("| Model | n_seeds | NDCG@10 | P@10 | R@10 | AUC-ROC | F1@0.5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if "status" in r:
            lines.append(f"| {r['model']} | MISSING | - | - | - | - | - |")
            continue
        def fmt(m):
            v = r.get(f"{m}_mean")
            s = r.get(f"{m}_std", 0.0)
            if v is None:
                return "-"
            if s and s > 0:
                return f"{v:.4f} ± {s:.4f}"
            return f"{v:.4f}"
        lines.append(
            f"| {r['model']} | {r['n_seeds']} | "
            f"{fmt('ndcg_at_10')} | {fmt('precision_at_10')} | "
            f"{fmt('recall_at_10')} | {fmt('auc_roc')} | {fmt('f1_at_0_5')} |"
        )

    if comparisons:
        lines.append("\n## Paired t-tests on AUC-ROC across 5 seeds\n")
        lines.append("| A | B | mean_diff (B-A) | t | p | Cohen's d | effect | sig @ 0.05 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in comparisons:
            if c["t_stat"] is None:
                continue
            lines.append(
                f"| {c['label_a']} | {c['label_b']} | "
                f"{c['mean_diff_b_minus_a']:+.4f} | "
                f"{c['t_stat']:+.3f} | {c['p_value']:.4f} | "
                f"{c['cohens_d']:+.3f} | {c['effect_size_bucket']} | "
                f"{'YES' if c['significant_at_alpha'] else 'no'} |"
            )

    missing = [n for n in RESULT_FILES if loaded[n] is None]
    if missing:
        lines.append(f"\n*Missing models (not yet run):* {', '.join(missing)}")

    (OUT_DIR / "MODEL_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
