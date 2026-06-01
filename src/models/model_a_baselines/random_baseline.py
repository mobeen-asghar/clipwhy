"""Random baseline: shuffle segments at random, compute ranking metrics.

Run N_TRIALS = 100 times with different seeds; report mean +/- std. This
establishes the absolute floor that every trained model must beat.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation import metrics as eval_metrics
from src.models.shared.data import load_split

N_TRIALS = 100
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("clipwhy.model_a.random")


def run(split: str = "test") -> dict:
    data = load_split(split)
    log.info("Random baseline: %s split, %d segments, %d positives",
             split, len(data.y), int(data.y.sum()))

    trial_results = []
    for i in range(N_TRIALS):
        rng = np.random.default_rng(seed=i)
        y_score = rng.random(size=data.y.shape).astype(np.float32)
        r = eval_metrics.evaluate(
            data.y, y_score, data.video_ids,
            split=split, model="random", seed=i,
        )
        trial_results.append(r)

    agg = eval_metrics.aggregate_seed_results(trial_results)
    agg["n_trials"] = N_TRIALS

    out = RESULTS_DIR / f"random_baseline_{split}.json"
    out.write_text(json.dumps(agg, indent=2))
    log.info("Wrote %s", out)

    print(f"\n== Random baseline ({split}, {N_TRIALS} trials) ==")
    for metric in ["ndcg_at_10", "precision_at_10", "recall_at_10", "auc_roc", "f1_at_0_5"]:
        m = agg[metric]
        print(f"  {metric:18s}: mean={m['mean']:.4f} std={m['std']:.4f}")
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run("test")
