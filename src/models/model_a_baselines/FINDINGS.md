# Model A — Baselines (ClipWhy V2)

Results on the test split (57,962 segments, 557 videos, 504 positive segments
at the V1-strict 2.0x / 3.0x thresholds). These baselines establish the
performance floor; every trained model must beat them to justify its cost.

## Results (test split)

| Model | Trials | NDCG@10 | P@10 | R@10 | AUC-ROC | F1@0.5 |
|---|---|---|---|---|---|---|
| Random (100 shuffles) | 100 | 0.097 ± 0.010 | 0.030 ± 0.003 | 0.187 ± 0.017 | 0.500 ± 0.013 | 0.017 ± 0.001 |
| Rule-based (deterministic) | 1 | 0.105 | 0.032 | 0.199 | 0.520 | 0.018 |

## Observations

- Random produces AUC-ROC 0.500, confirming the evaluation harness is correctly
  bisecting positives and negatives (random = chance by definition).
- Rule-based beats random by a small margin on every metric. The absolute gap
  is small because the V1 feature set the rule was designed for no longer
  exists in V2:
  - V1's rule used `emotional_intensity` and `sentiment_arc_range` (VADER).
  - V2 dropped all VADER features (they added noise per V1 ablation).
  - We swapped them for `arousal_mean` and `arousal_arc_direction` from
    Wav2Vec 2.0. These measure a genuinely different signal; the V1 weights
    that worked for VADER don't necessarily transfer.
- This is useful evidence for the FPR: the V1 rule-based winner was an artefact
  of the small-data regime. Once the feature landscape changes (and the
  training population broadens to 394 creators), hand-coded weights don't hold.

## Reproducing

```bash
python -m src.models.model_a_baselines.run --split test
```

Writes `results/random_baseline_test.json` and `results/rule_based_test.json`.

## Rule weights (unchanged structure from V1)

| Feature | Weight |
|---|---|
| hook_word_ratio | 0.20 |
| energy_mean | 0.20 |
| arousal_mean (was: emotional_intensity) | 0.15 |
| words_per_second | 0.15 |
| arousal_arc_direction (was: sentiment_arc_range) | 0.10 |
| is_intro | 0.10 |
| speaking_rate_audio | 0.10 |

Re-tuning the weights to V2 features is a future-work item. The point of this
baseline is a naive port from V1 so the RQ2 comparison is meaningful.
