"""Paired t-tests and Cohen's d for comparing two models across seeds.

Used in FINDINGS to answer RQ2 (ML vs rule-based) and RQ3 (multimodal vs
single-modality) with statistical rigour (alpha=0.05).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_rel


def cohens_d(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Paired Cohen's d on differences (b - a).

    Interpretation (absolute value):
      < 0.2  negligible
      < 0.5  small
      < 0.8  medium
      >= 0.8 large
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 0 else 0.0


def compare(
    model_a_scores: list[float],
    model_b_scores: list[float],
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
) -> dict:
    """Paired t-test plus Cohen's d.

    Null: mean(B - A) == 0. If p < alpha and mean diff > 0, B significantly
    outperforms A. Cohen's d quantifies the size of that effect.
    """
    a = np.asarray(model_a_scores, dtype=float)
    b = np.asarray(model_b_scores, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return {"n_pairs": len(a), "t_stat": None, "p_value": None,
                "cohens_d": None, "significant": False, "direction": None,
                "label_a": label_a, "label_b": label_b}
    t, p = ttest_rel(a, b)
    d = cohens_d(a, b)
    mean_diff = float(b.mean() - a.mean())
    direction = label_b if mean_diff > 0 else label_a if mean_diff < 0 else None
    return {
        "label_a": label_a, "label_b": label_b,
        "n_pairs": int(len(a)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff_b_minus_a": mean_diff,
        "t_stat": float(t), "p_value": float(p),
        "cohens_d": d,
        "effect_size_bucket": _bucket(abs(d)),
        "significant_at_alpha": bool(p < alpha),
        "alpha": float(alpha),
        "direction": direction,
    }


def _bucket(abs_d: float) -> str:
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"
