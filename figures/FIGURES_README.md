# Regenerated figures (ViralPredict / ClipWhy)

All figures regenerated from the original result files at **600 DPI**.
Each is provided as **PNG** (600 DPI raster) and **PDF** (vector, scales losslessly).
The chart titles deliberately contain **no figure number** (the numbered caption
lives in the Word document, per supervisor feedback).

Regenerate with:

```
clipwhy-demo/venv/bin/python paper-publication/regenerate_figures.py
```

## Mapping to the paper

| Paper figure | File | What it shows | Data source |
|---|---|---|---|
| Figure 1 | `fig1_positives_per_category` | Positive segments per content category, Exp 1 and Exp 2 | segment labels (both experiments) |
| Figure 2 | `fig2_exp1_learning_curve` | Exp 1 learning curve, RF and XGBoost AUC-ROC vs training positives, **with error bars (std over 5 seeds)** | `clipwhy-pipeline/output/results/scaling_experiment.json` |
| Figure 3 | `fig3_exp1_vs_exp2_auc` | Exp 1 vs Exp 2 AUC-ROC per matched model architecture | aggregated model metrics, both experiments |
| Figure 4 | `fig4_exp2_spearman_top30` | Top-30 Exp 2 features by absolute Spearman correlation with the label | `segments_with_splits.csv` (full set) |
| Figure 5 | `fig5_exp2_position_ratio_hist` | position_ratio histogram, positive vs negative (Exp 2 training split) | `segments_with_splits.csv` (train split) |
| Figure 6 | `fig6_exp1_ablation_remove` | Exp 1 remove-one-category ablation (Random Forest) | `clipwhy-pipeline/output/results/ablation_study.json` |
| Figure 7 | `fig7_exp1_single_category` | Exp 1 single-category isolation (Random Forest) | `ablation_study.json` |
| Figure 8 | `fig8_exp2_ablation_remove` | Exp 2 remove-one-category ablation (XGBoost) | `clipwhy-v2/.../ablation_xgboost.json` |
| Figure 9 | `fig9_exp2_single_category` | Exp 2 single-category isolation (XGBoost) | `ablation_xgboost.json` |

## Note on Figure 4

The top four features are all **structural** (`video_duration`, `is_first_segment`,
`position_ratio`, `is_intro`); rank 5 is `largest_face_area_ratio_max` (a **visual**
feature). No CLIP PCA dimension appears in the top 30. This means the Discussion
sentence "the top five features are all structural or creator-context indicators"
should read **"the top four features are all structural"** to match the figure.
