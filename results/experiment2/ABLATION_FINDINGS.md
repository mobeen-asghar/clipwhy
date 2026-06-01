# Per-Category Ablation Study (XGBoost, V2)

Baseline XGBoost AUC: **0.7153 +/- 0.0000** (84 features, 5 seeds)

## Remove-One-Category (importance = how much AUC drops)

| Category | Features removed | AUC after | Δ AUC | t-stat | p-value | Cohen's d | Significant? |
|---|---|---|---|---|---|---|---|
| structural | 76 kept (8 removed) | 0.6625 +/- 0.0000 | -0.0527 | +inf | 0.0000 | +0.00 | YES |
| visual | 43 kept (41 removed) | 0.7020 +/- 0.0000 | -0.0133 | +inf | 0.0000 | +0.00 | YES |
| creator_context | 79 kept (5 removed) | 0.7113 +/- 0.0000 | -0.0039 | +inf | 0.0000 | +0.00 | YES |
| audio_emotion | 77 kept (7 removed) | 0.7123 +/- 0.0000 | -0.0030 | +inf | 0.0000 | +0.00 | YES |
| voice_quality | 82 kept (2 removed) | 0.7166 +/- 0.0000 | +0.0013 | -inf | 0.0000 | +0.00 | YES |
| audio_speech | 77 kept (7 removed) | 0.7178 +/- 0.0000 | +0.0025 | -inf | 0.0000 | +0.00 | YES |
| audio_events | 80 kept (4 removed) | 0.7210 +/- 0.0000 | +0.0057 | -inf | 0.0000 | +0.00 | YES |
| text | 74 kept (10 removed) | 0.7216 +/- 0.0000 | +0.0063 | -inf | 0.0000 | +0.00 | YES |

## Single-Category-Isolation (only that category's features)

| Category | Features used | AUC | Δ vs full | Above random (0.50)? |
|---|---|---|---|---|
| structural | 8 | 0.6971 +/- 0.0000 | -0.0181 | YES |
| visual | 41 | 0.6420 +/- 0.0000 | -0.0733 | YES |
| audio_speech | 7 | 0.5571 +/- 0.0000 | -0.1582 | YES |
| text | 10 | 0.5567 +/- 0.0000 | -0.1586 | YES |
| audio_events | 4 | 0.5450 +/- 0.0000 | -0.1702 | barely |
| audio_emotion | 7 | 0.5447 +/- 0.0000 | -0.1706 | barely |
| creator_context | 5 | 0.5333 +/- 0.0000 | -0.1820 | barely |
| voice_quality | 2 | 0.5287 +/- 0.0000 | -0.1866 | barely |
