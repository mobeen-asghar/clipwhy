"""Audio emotion features (7) via Wav2Vec 2.0.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim.
Output per forward: (arousal, dominance, valence), each in ~[0, 1].

Mean features: from one forward pass over the full 30-s clip.
Dynamics features (std / peak / arc): from a sliding-window pass over 5-s
windows at 2.5-s hop (12 windows per 30 s).
"""
import logging
from typing import Optional

import librosa
import numpy as np

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.audio_emotion")

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

_ZERO = {
    "arousal_mean": 0.0, "valence_mean": 0.0, "dominance_mean": 0.0,
    "arousal_std": 0.0, "arousal_peak": 0.0,
    "arousal_arc_direction": 0.0, "valence_arc_direction": 0.0,
}


def load_wav2vec_emotion(device: str = "cuda"):
    """Load the audeering dimensional emotion model.

    The model card on HuggingFace ships a custom `EmotionModel` class; we
    replicate the minimal version here so we don't depend on their repo.
    """
    try:
        import torch
        import torch.nn as nn
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
    except ImportError as e:
        log.error("transformers / torch not installed: %s", e)
        return None

    class RegressionHead(nn.Module):
        def __init__(self, config_):
            super().__init__()
            self.dense = nn.Linear(config_.hidden_size, config_.hidden_size)
            self.dropout = nn.Dropout(config_.final_dropout)
            self.out_proj = nn.Linear(config_.hidden_size, 3)

        def forward(self, x):
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config_):
            super().__init__(config_)
            self.wav2vec2 = Wav2Vec2Model(config_)
            self.classifier = RegressionHead(config_)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            pooled = hidden.mean(dim=1)
            logits = self.classifier(pooled)
            return pooled, logits

    try:
        # NOTE: the audeering model is audio-only (dimensional regression).
        # It has no tokenizer/vocab, so using Wav2Vec2Processor (which bundles
        # feature_extractor + tokenizer) fails on newer transformers that
        # validate vocab_size strictly. Wav2Vec2FeatureExtractor works directly.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = EmotionModel.from_pretrained(MODEL_NAME)
        model.eval()
        if device == "cuda":
            model = model.cuda()
            # Enable FP16 for inference speed
            model = model.half()
        return {"model": model, "processor": feature_extractor, "device": device}
    except Exception as e:
        log.error("wav2vec emotion load failed: %s", e)
        return None


def extract(job: SegmentJob, models: dict) -> dict:
    bundle = models.get("wav2vec_emotion")
    if bundle is None:
        return dict(_ZERO)

    try:
        y, sr = librosa.load(str(job.audio_path), sr=16000)
    except Exception as e:
        log.warning("audio_emotion load failed for %s: %s", job.segment_id, e)
        return dict(_ZERO)
    if len(y) == 0:
        return dict(_ZERO)

    import torch
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]

    win = int(config.WAV2VEC_WINDOW_SEC * 16000)
    hop = int(config.WAV2VEC_HOP_SEC * 16000)

    # ── Build the batch: full segment + all 5-s windows ─────────────────────
    # Old code did 13 separate forward passes. Now we stack the full segment
    # (variable length) and a batch of fixed-length windows (all same length
    # so no padding needed) into 2 forward passes total. 6-7x speedup on this
    # stage at zero quality cost.
    if len(y) >= win:
        window_clips = [y[s : s + win] for s in range(0, len(y) - win + 1, hop)]
    else:
        window_clips = [y]

    def _infer_one(wav: np.ndarray) -> np.ndarray:
        """One forward, returns [arousal, dominance, valence]."""
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"]
        if device == "cuda":
            input_values = input_values.cuda().half()
        with torch.no_grad():
            _, logits = model(input_values)
        return logits.squeeze(0).float().cpu().numpy()

    def _infer_batch(clips: list) -> np.ndarray:
        """Batched forward over equal-length clips. Returns (N, 3)."""
        # All clips are the same length (win samples) so we can stack directly.
        batched = np.stack(clips, axis=0).astype(np.float32)
        inputs = processor(
            list(batched), sampling_rate=16000, return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"]
        if device == "cuda":
            input_values = input_values.cuda().half()
        with torch.no_grad():
            _, logits = model(input_values)
        return logits.float().cpu().numpy()  # (N, 3)

    # Full-segment pass (variable length, 1 forward).
    full = _infer_one(y)
    arousal_mean, dominance_mean, valence_mean = float(full[0]), float(full[1]), float(full[2])

    # Windowed pass (12 windows in a single batched forward).
    if window_clips:
        windowed = _infer_batch(window_clips)
    else:
        windowed = full.reshape(1, 3)

    a_win = windowed[:, 0]
    v_win = windowed[:, 2]
    arousal_std = float(np.std(a_win))
    arousal_peak = float(np.max(a_win))

    # Arc direction: last 5 s mean minus first 5 s mean. In window units, with
    # 2.5-s hop this is ~2 windows at each end.
    k = max(1, min(2, len(a_win) // 3))
    arousal_arc_direction = float(np.mean(a_win[-k:]) - np.mean(a_win[:k]))
    valence_arc_direction = float(np.mean(v_win[-k:]) - np.mean(v_win[:k]))

    return {
        "arousal_mean": round(arousal_mean, 6),
        "valence_mean": round(valence_mean, 6),
        "dominance_mean": round(dominance_mean, 6),
        "arousal_std": round(arousal_std, 6),
        "arousal_peak": round(arousal_peak, 6),
        "arousal_arc_direction": round(arousal_arc_direction, 6),
        "valence_arc_direction": round(valence_arc_direction, 6),
    }
