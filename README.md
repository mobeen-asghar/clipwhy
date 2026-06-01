# ClipWhy

**Explainable, segment-level virality prediction for long-form video.**

ClipWhy takes a long-form video (podcast, interview, lecture) and ranks its
30-second segments by how likely each is to succeed as a creator-repurposed
short, using only **pre-publication, content-level features**. Every prediction
is backed by an auditable 84-feature schema, so a score can be explained rather
than handed down as a black box.

This repository is the research pipeline behind the paper:

> M. M. Asghar and M. Helal, *ViralPredict: segment-level virality prediction
> for short-form video* (ICaTAS 2026).

It contains the full feature-extraction pipeline, the four model families
compared in the paper, the evaluation and ablation code, the result files behind
every table, and the scripts that regenerate every figure.

---

## Key results

Test-set ranking on **394 creators, 3,709 long-form videos, 389,619 segments,
3,263 viral positives** (Experiment 2), AUC-ROC, mean over 5 seeds:

| Model | AUC-ROC | NDCG@10 | Recall@10 |
|---|---|---|---|
| Random | 0.500 | 0.097 | 0.187 |
| Rule-based | 0.520 | 0.105 | 0.199 |
| Random Forest | 0.633 | 0.147 | 0.240 |
| BERT (text) | 0.635 | 0.148 | 0.278 |
| Multi-modal, 3-branch | 0.682 | 0.143 | 0.254 |
| XGBoost | 0.716 | 0.158 | 0.243 |
| **Multi-modal, 4-branch** | **0.739** | **0.203** | **0.336** |

Main findings:

- **Structural position dominates** at 30-second granularity: 8 structural
  features alone reach AUC 0.697, which is 97% of the full 84-feature model.
- A **four-branch multi-modal architecture** (text + audio + visual + an explicit
  metadata branch) is the best model, significantly beating XGBoost, BERT and the
  three-branch variant (largest effect size d = 21.9 for the metadata branch).
- The advantage of learned models over a rule-based heuristic **emerges with
  scale**: at 84 positives the rule-based baseline wins, but it is overtaken as
  the data grows toward 2,277 positives.

Numbers in this table are reproduced verbatim from `results/`.

---

## Architecture

```
Video in -> Ingestion -> Feature extraction -> Model -> Score + explanation
```

**84 features in 8 categories** (full machine-readable list in
`src/feature_extraction/feature_catalog.json`):

| Category | # | Extractor |
|---|---|---|
| Text | 10 | hook lexicon (129 words), Whisper transcript stats |
| Audio (speech) | 7 | librosa (energy, pitch, speaking rate, silence) |
| Voice quality | 2 | Praat / parselmouth (jitter, shimmer) |
| Audio events | 4 | YAMNet + laughter detector (music, laughter) |
| Audio emotion | 7 | Wav2Vec 2.0 (arousal, valence, dominance) |
| Visual | 41 | CLIP (PCA-32), DOVER quality, TransNetV2 cuts, InsightFace |
| Structural | 8 | position, intro/outro, duration, novelty |
| Creator context | 5 | one-hot content category |

**Four model families** compared:

- **Baselines** (`model_a_baselines`) - random and a hand-weighted rule-based score
- **Traditional ML** (`model_b_traditional_ml`) - Random Forest, XGBoost
- **BERT** (`model_c_bert`) - `bert-base-uncased` fine-tuned on transcripts
- **Multi-modal** (`model_d_multimodal`) - late fusion, 3-branch and 4-branch

---

## Repository structure

```
clipwhy/
├── src/
│   ├── data_collection/      # creator discovery + pairing (manual seed)
│   ├── data_collection_v2/   # automated creator discovery (394 creators)
│   ├── feature_extraction/   # 84-feature pipeline + extractors + feature_catalog.json
│   ├── post_extraction/      # merge, relabel, split, CLIP PCA, normalise
│   ├── models/               # model_a..d + compare.py (statistical tests)
│   └── evaluation/           # metrics, ablation, statistical tests
├── config/                   # settings, thresholds
├── results/
│   ├── experiment2/          # 84-feature pipeline (paper's primary results)
│   └── experiment1/          # 35-feature pilot (learning-curve + ablation)
├── figures/                  # the 9 paper figures (600 DPI PNG + vector PDF) + regenerate script
├── requirements.txt          # data collection
├── requirements_features.txt # feature extraction (GPU)
└── run.sh
```

> This repository holds the **Experiment 2** pipeline (84 features), the paper's
> primary system. Experiment 1 was a smaller 35-feature pilot (15 creators); its
> result files are included under `results/experiment1/` for the learning-curve
> and ablation comparisons.

---

## Installation

Requirements: **Python 3.10+**, **ffmpeg**, and (for the GPU feature extractors
and BERT) a CUDA GPU.

```bash
git clone https://github.com/mobeen-asghar/clipwhy.git
cd clipwhy

python3 -m venv venv
source venv/bin/activate

# Data collection + lightweight steps
pip install -r requirements.txt

# Feature extraction (GPU: CLIP, DOVER, Wav2Vec 2.0, YAMNet, ...)
pip install -r requirements_features.txt
```

The pipeline reads and writes through environment variables (YouTube API keys,
object storage, optional Discord notifications). Copy `.env.example` to `.env`
and fill in your own values; no credentials are committed to this repository.

---

## Example usage

All commands are run from the repository root with `PYTHONPATH=.`.

```bash
# 1. Collect data: discover creators, pair shorts to source long videos
python -m src.data_collection_v2.cli run --vm-id gpu0

# 2. Extract the 84 features per 30-second segment
python -m src.feature_extraction.cli run --vm-id gpu0 --device cuda

# 3. Post-process: merge, label, split (70/15/15 by video), CLIP PCA, normalise
python -m src.post_extraction.cli all

# 4. Train and evaluate the models
python -m src.models.model_a_baselines.run            # random + rule-based
python -m src.models.model_b_traditional_ml.run --model both   # RF + XGBoost
python -m src.models.model_c_bert.run                 # BERT
python -m src.models.model_d_multimodal.run           # 3-branch + 4-branch

# 5. Compare models (paired t-tests, Cohen's d) and run the ablation study
python -m src.models.compare
python -m src.evaluation.ablation_study

# 6. Regenerate every paper figure (600 DPI PNG + vector PDF)
python figures/regenerate_figures.py
```

Add `--help` to any command for its options (for example, the labelling
thresholds: `post_extraction relabel --eng-mult 2.0 --vpd-mult 3.0`).

---

## Data availability

The dataset (raw videos, segment media, and feature matrices) is **not included**
because of its size (over a terabyte of media) and YouTube's terms of service. The
pipeline reconstructs it from public YouTube content given the creator list and a
YouTube Data API key. The small result files needed to reproduce every table and
figure in the paper are included under `results/` and `figures/`.

A clip is labelled **viral** if, relative to the creator's own historical
baseline, its engagement rate exceeds 2.0x the median **or** its views-per-day
exceed 3.0x the median. Train/validation/test splits are **by video** (70/15/15,
`random_state=42`) to prevent leakage between segments of the same video.

---

## Citation

```bibtex
@inproceedings{asghar2026viralpredict,
  title     = {ViralPredict: Segment-Level Virality Prediction for Short-Form Video},
  author    = {Asghar, Muhammad Mobeen and Helal, Manal},
  booktitle = {Proceedings of ICaTAS 2026},
  year      = {2026}
}
```
(Update the title, venue and pages once the paper is finalised.)

## License

Released under the MIT License. See [LICENSE](LICENSE).
