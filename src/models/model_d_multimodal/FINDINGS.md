# Model D — Multimodal Fusion (ClipWhy V2)

Late-fusion neural network combining BERT [CLS] + audio + visual + (optional)
metadata branches. Two architectures evaluated, both with 5 seeds.

Training on RunPod A100 SXM 80GB. Fine-tuned BERT seed=42 used as a frozen
feature extractor (CLS embeddings precomputed once, then 5 seeds of MLP
training).

## Results (test split, 5 seeds each)

| Variant | Branches | Inputs | NDCG@10 | P@10 | R@10 | **AUC-ROC** | F1@0.5 |
|---|---|---|---|---|---|---|---|
| **original** (V1-style) | 3 | text + audio + visual | 0.143 +/- 0.004 | 0.042 +/- 0.001 | 0.254 +/- 0.012 | **0.6815 +/- 0.0052** | 0.053 +/- 0.002 |
| **with_metadata** (V2-improved) | 4 | + structural + creator_context | 0.203 +/- 0.006 | 0.052 +/- 0.002 | 0.336 +/- 0.014 | **0.7385 +/- 0.0031** | 0.070 +/- 0.005 |

Per-seed AUC for `with_metadata`: 0.7419, 0.7375, 0.7348, 0.7414, 0.7368
Per-seed AUC for `original`:      0.6850, 0.6863, 0.6776, 0.6790, 0.6798

## The headline finding: metadata branch matters enormously

Adding a 4th branch for structural + creator_context features lifts AUC
from 0.6815 to 0.7385, a +0.057 jump. Paired t-test:

```
mean_diff = +0.0569
t = -48.866
p < 0.0001
Cohen's d = 21.85 (massive effect)
```

**Why the original V1-style architecture underperforms:** XGBoost's top-5
features by importance are `video_duration`, `is_first_segment`,
`creator_category_commentary`, `creator_category_entertainment`,
`position_ratio`. All five live in the structural + creator_context
categories. The 3-branch design excludes these entirely, expecting the
text+audio+visual representations to compensate. They don't, by a lot.

The 4-branch design treats metadata as a fourth modality (engineered
numeric features projected through a small MLP) and lets the fusion layer
learn how to weight it against the learned representations. Fusion learns
the metadata branch carries roughly half the discriminative signal.

## Compared to V1

| Model | V1 AUC (84 pos) | V2 AUC (2,277 pos) |
|---|---|---|
| Multimodal (V1, audio + BERT only) | 0.472 +/- ?? | 0.6815 +/- 0.0052 |
| Multimodal_with_metadata (new) | n/a | **0.7385 +/- 0.0031** |

V1's multimodal underperformed even rule-based at small scale (a known
fusion problem with few positives). V2's 3-branch already comfortably
beats every V1 model, and the 4-branch beats every V2 model.

## Compared to other V2 models

| Model | AUC | vs multimodal_with_metadata |
|---|---|---|
| Random | 0.500 | -0.239 (p<0.001, large effect) |
| Rule-based | 0.520 | -0.219 (n=1, no t-test) |
| Random Forest | 0.633 | -0.106 (p<0.001, d=10.8) |
| BERT | 0.635 | -0.103 (p<0.001, d=11.1) |
| XGBoost | 0.716 | -0.023 (p=0.0001, d=7.4) |
| **Multimodal_with_metadata** | **0.7385** | new V2 champion |

Multimodal_with_metadata is statistically significantly better than every
other model, including the previous best XGBoost.

## Ranking metrics — the SaaS-relevant numbers

For each video in the test set, top-K precision/recall/NDCG averaged across
videos that have >=1 positive segment (297/557 videos qualify).

| Metric | XGBoost | Multimodal_with_metadata | Lift |
|---|---|---|---|
| NDCG@10 | 0.158 | 0.203 | +28% |
| Precision@10 | 0.039 | 0.052 | +33% |
| Recall@10 | 0.243 | 0.336 | +38% |
| F1@0.5 | 0.038 | 0.070 | +84% |

Recall@10 went from "finds 1 in 4 viral segments in top 10" to "finds 1 in
3". For SaaS UX (give the user 10 candidate clips, of which they pick 5),
this is the metric that matters most. Combined with an LLM re-ranker on
top of the candidate set, this becomes the SaaS-shippable foundation.

## Reproducing

```bash
# Pre-reqs:
#   - Post-extraction complete (data/post_extraction/feature_matrix_*.csv)
#   - Model C trained (results/bert_checkpoints/bert_seed42/) OR pull from R2

# 3-branch original (V1-style)
python -m src.models.model_d_multimodal.multimodal --variant original --bert-seed 42

# 4-branch with metadata (V2-improved)
python -m src.models.model_d_multimodal.multimodal --variant with_metadata --bert-seed 42
```

Outputs land in `results/`:
- `multimodal_original_seed{N}_test.json` + `multimodal_original_test.json`
- `multimodal_with_metadata_seed{N}_test.json` + `multimodal_with_metadata_test.json`
- `multimodal_checkpoints/` (small ~770 KB per seed, gitignored)

R2 backup: `r2:clipwhy-data/models/multimodal_original/` and
`r2:clipwhy-data/models/multimodal_with_metadata/`.

## Hyperparameters

| Parameter | Value |
|---|---|
| Fusion | Late (per-modality MLP, then concat, then small MLP head) |
| Branches | original: 3 (text 768->128, audio 20->128, visual 42->128, fuse 384->128->2) |
| | with_metadata: 4 (above + metadata 13->64->128, fuse 512->128->2) |
| BERT | Frozen (Model C seed=42 checkpoint), CLS extracted once per split |
| MLP learning rate | 1e-4 (AdamW, weight_decay=0.01) |
| Batch size | 32 train / 128 eval |
| Epochs | up to 5 with early stopping on val AUC-ROC (patience=1) |
| Loss | Cross-entropy with class weights (pos = neg/pos = ~120) |
| Seeds | 42, 123, 456, 789, 1024 |
| GPU | A100 SXM 80GB |
| Wall clock per seed | ~5 min (post-CLS-extraction) |

## Total parameters

| Variant | Params | Branches |
|---|---|---|
| original | 174,082 | 3 |
| with_metadata | 194,242 | 4 |

Tiny by deep learning standards. The work happens upstream in BERT
(frozen, 110M params) and the engineered features themselves. The fusion
MLP just learns how to weight them.
