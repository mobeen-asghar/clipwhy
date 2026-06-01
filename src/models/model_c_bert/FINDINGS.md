# Model C — BERT (ClipWhy V2)

Fine-tuned `bert-base-uncased` on segment transcripts for binary viral
classification. 5 seeds, AdamW @ 2e-5, batch 32, max_length 256, up to 3
epochs with early stopping on val AUC-ROC.

Training executed on RunPod A100 SXM 80GB (~10h wall clock total, fp32).

## Results (test split, 5 seeds)

| Metric | Mean | Std |
|---|---|---|
| AUC-ROC | **0.6354** | 0.0067 |
| NDCG@10 | 0.1477 | 0.0122 |
| Precision@10 | 0.0442 | 0.0028 |
| Recall@10 | 0.2780 | 0.0220 |
| F1@0.5 | 0.0000 | 0.0000 |

Per-seed test AUC: 0.6314, 0.6307, 0.6414, 0.6296, 0.6440

## Observations

**BERT plateaus around AUC 0.635.** Tighter cross-seed std than V1 (V1 had
±0.036 with 84 positives; V2 with 2,277 positives gives ±0.007). The model
is converging on the same answer regardless of initialisation, which is the
right diagnostic: more data didn't lift BERT, it just made the model more
confident in its plateau.

**Severe overfitting, every seed.** Val AUC peaks at epoch 0 (0.65 range)
and degrades each subsequent epoch:

| Seed | epoch 0 val | epoch 1 val | epoch 2 val |
|---|---|---|---|
| 42 | 0.6507 | 0.6190 | 0.5912 |
| 123 | 0.6408 | 0.5996 | 0.5908 |
| 456 | 0.6559 | 0.6331 | 0.6039 |
| 789 | 0.6431 | 0.5766 | 0.6212 |
| 1024 | 0.6500 | 0.6112 | 0.5950 |

The early-stop mechanism preserves the epoch-0 weights as the best
checkpoint, so test results use the right model. The training loop still
runs all 3 epochs because the patience-> 1 condition needs 2 consecutive
worse epochs before breaking, but no test-time loss.

**F1@0.5 = 0.000 means the calibration is off.** BERT outputs are heavily
weighted toward "not viral" (the model rarely produces probability >= 0.5
even for true positives). This is a calibration problem, not a ranking
problem. AUC-ROC is rank-based and unaffected.

## Compared to V1

| Model | V1 AUC (84 pos) | V2 AUC (2,277 pos) | Lift |
|---|---|---|---|
| BERT | 0.566 ± 0.036 | 0.635 ± 0.007 | **+0.069** |

Modest absolute gain, large gain in precision (much tighter std). V2 has
27x more positive training examples but BERT's ceiling appears to be
fundamentally bounded by what can be learned from transcript text alone.

## Compared to other V2 models

| Model | AUC | Significantly different from BERT? |
|---|---|---|
| Random Forest | 0.633 ± 0.010 | no (p=0.69) |
| XGBoost | 0.716 ± 0.000 | yes, XGB much better (p<0.001, d=-12) |
| Multimodal (with metadata) | 0.7385 ± 0.0031 | yes, multimodal much better (p<0.001, d=11) |

**Key finding: BERT and Random Forest are statistically tied.** Text
semantics alone get you to the same place that 84 engineered features in
a tree ensemble get you. This validates V1's intuition that engineered
features are sufficient at this scale and confirms it with stronger
statistical power at V2 scale.

## Reproducing

```bash
# Pre-req: post-extraction complete
python -m src.models.model_c_bert.prepare_data
python -m src.models.model_c_bert.bert_model
```

Outputs:
- `results/bert_seed{42,123,456,789,1024}_test.json` (per seed)
- `results/bert_test.json` (aggregated)
- `results/bert_checkpoints/bert_seed{N}/` (HuggingFace save_pretrained dump,
  ~440 MB each, gitignored)

Fine-tuned BERT seed=42 checkpoint also pushed to `r2:clipwhy-data/models/bert_seed42/`
for use as the frozen feature extractor in Model D.

## Hyperparameters used

| Parameter | Value |
|---|---|
| Base model | bert-base-uncased |
| Max length | 256 tokens |
| Batch size (train/eval) | 32 / 128 |
| Learning rate | 2e-5 |
| Optimizer | AdamW, weight_decay=0.01 |
| Warmup steps | 500 |
| Epochs | up to 3, early stop patience=1 |
| Loss | Cross-entropy with class weights (pos = neg/pos = ~120) |
| Seeds | 42, 123, 456, 789, 1024 |
| Precision | fp32 |
| GPU | A100 SXM 80GB (single) |
