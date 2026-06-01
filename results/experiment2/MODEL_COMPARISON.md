# Model Comparison (test split)

| Model | n_seeds | NDCG@10 | P@10 | R@10 | AUC-ROC | F1@0.5 |
|---|---|---|---|---|---|---|
| random | 100 | 0.0968 ± 0.0103 | 0.0297 ± 0.0026 | 0.1867 ± 0.0171 | 0.5002 ± 0.0130 | 0.0171 ± 0.0008 |
| rule_based | 1 | 0.1049 | 0.0320 | 0.1988 | 0.5195 | 0.0182 |
| rf | 5 | 0.1474 ± 0.0043 | 0.0375 ± 0.0022 | 0.2399 ± 0.0182 | 0.6328 ± 0.0098 | 0.0000 |
| xgboost | 5 | 0.1576 | 0.0391 | 0.2428 | 0.7157 | 0.0380 |
| bert | 5 | 0.1477 ± 0.0122 | 0.0442 ± 0.0028 | 0.2780 ± 0.0220 | 0.6354 ± 0.0067 | 0.0000 |
| multimodal | MISSING | - | - | - | - | - |
| multimodal_original | 5 | 0.1432 ± 0.0036 | 0.0422 ± 0.0014 | 0.2537 ± 0.0116 | 0.6815 ± 0.0052 | 0.0526 ± 0.0020 |
| multimodal_with_metadata | 5 | 0.2032 ± 0.0062 | 0.0523 ± 0.0021 | 0.3356 ± 0.0137 | 0.7385 ± 0.0031 | 0.0700 ± 0.0045 |

## Paired t-tests on AUC-ROC across 5 seeds

| A | B | mean_diff (B-A) | t | p | Cohen's d | effect | sig @ 0.05 |
|---|---|---|---|---|---|---|---|
| rf | xgboost | +0.0829 | -19.005 | 0.0000 | +8.499 | large | YES |
| rf | bert | +0.0026 | -0.422 | 0.6949 | +0.189 | negligible | no |
| rf | multimodal_original | +0.0487 | -12.066 | 0.0003 | +5.396 | large | YES |
| rf | multimodal_with_metadata | +0.1057 | -24.168 | 0.0000 | +10.808 | large | YES |
| xgboost | bert | -0.0803 | +26.729 | 0.0000 | -11.953 | large | YES |
| xgboost | multimodal_original | -0.0341 | +14.739 | 0.0001 | -6.592 | large | YES |
| xgboost | multimodal_with_metadata | +0.0228 | -16.537 | 0.0001 | +7.396 | large | YES |
| bert | multimodal_original | +0.0461 | -9.101 | 0.0008 | +4.070 | large | YES |
| bert | multimodal_with_metadata | +0.1031 | -24.856 | 0.0000 | +11.116 | large | YES |
| multimodal_original | multimodal_with_metadata | +0.0569 | -48.866 | 0.0000 | +21.853 | large | YES |

*Missing models (not yet run):* multimodal
