"""Audio event features (4): YAMNet music + speech ratio + Gillick laughter.

YAMNet is the TF-Hub audio-event classifier (521 AudioSet classes).
We use a PyTorch port (torch_audioset) to avoid pulling TensorFlow into the
CLIP+DOVER+Wav2Vec stack.

Gillick laughter-detection is a separate small CNN specifically for laughter
recall; F1 0.75-0.80 vs YAMNet's generic Laughter class at 0.55-0.65.
"""
import logging
from typing import Optional

import librosa
import numpy as np

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.audio_events")

_ZERO = {
    "music_presence": 0.0,
    "music_fraction": 0.0,
    "speech_music_ratio": 0.0,
    "laughter_peak": 0.0,
}


# ── Model loading ───────────────────────────────────────────────────────────
def load_yamnet(device: str = "cuda"):
    """Load YAMNet via the torch_audioset port.

    torch_audioset's YAMNet expects preprocessed mel-spectrogram input, not
    raw waveform. We pair the model with WaveformToInput to handle the
    feature-extraction step.
    """
    try:
        import torch
        from torch_audioset.yamnet.model import yamnet as yamnet_model
        from torch_audioset.yamnet.model import yamnet_category_metadata
        from torch_audioset.data.torch_input_processing import WaveformToInput
    except ImportError as e:
        log.error("torch_audioset not installed: %s", e)
        return None
    try:
        model = yamnet_model(pretrained=True)
        model.eval()
        to_input = WaveformToInput()
        if device == "cuda":
            model = model.cuda()
            to_input = to_input.cuda()
        return {
            "model": model,
            "to_input": to_input,
            "meta": yamnet_category_metadata(),
        }
    except Exception as e:
        log.error("yamnet load failed: %s", e)
        return None


def load_gillick_laughter(device: str = "cuda"):
    """Load Gillick 2021 laughter-detection model + its mel feature extractor.

    Expects the jrgillick/laughter-detection repo cloned to
    config.LOCAL_MODELS / "laughter-detection" (done by setup_features_pod.sh).

    The Gillick repo is structured for `cwd=repo-root` usage: configs.py does
    `import models, audio_utils`, and `audio_utils.py` lives under `utils/`.
    Both the root AND the utils/ subdir need to be on sys.path.
    """
    try:
        import sys as _sys
        import torch
    except ImportError as e:
        log.warning("Gillick prereqs missing: %s", e)
        return None

    gillick_root = config.LOCAL_MODELS / "laughter-detection"
    if not gillick_root.exists():
        log.warning("Gillick repo not cloned at %s; laughter_peak will be 0", gillick_root)
        return None

    for p in (str(gillick_root), str(gillick_root / "utils")):
        if p not in _sys.path:
            _sys.path.insert(0, p)

    try:
        from configs import CONFIG_MAP  # from jrgillick repo root
    except Exception as e:
        log.warning("Gillick configs import failed (%s); laughter_peak will be 0", e)
        return None

    cfg_name = "resnet_with_augmentation"
    cfg = CONFIG_MAP[cfg_name]
    try:
        model = cfg["model"](
            dropout_rate=0.0,
            linear_layer_size=cfg["linear_layer_size"],
            filter_sizes=cfg["filter_sizes"],
        )
    except Exception as e:
        log.warning("Gillick model construction failed (%s); laughter_peak will be 0", e)
        return None

    ckpt = gillick_root / "checkpoints" / "in_use" / "resnet_with_augmentation" / "best.pth.tar"
    if not ckpt.exists():
        log.warning("Gillick checkpoint missing at %s; laughter_peak will be 0", ckpt)
        return None

    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        # Use set_device (Gillick's own method) so their custom layers register cuda.
        if hasattr(model, "set_device"):
            model.set_device(device)
        else:
            if device == "cuda":
                model = model.cuda()
        return {"model": model, "cfg": cfg}
    except Exception as e:
        log.warning("Gillick weights load failed (%s); laughter_peak will be 0", e)
        return None


# ── Extraction ──────────────────────────────────────────────────────────────
def extract(job: SegmentJob, models: dict) -> dict:
    yamnet_bundle = models.get("yamnet")
    gillick_bundle = models.get("gillick")

    try:
        y, sr = librosa.load(str(job.audio_path), sr=config.AUDIO_SAMPLE_RATE)
    except Exception as e:
        log.warning("audio_events load failed for %s: %s", job.segment_id, e)
        return dict(_ZERO)

    out = dict(_ZERO)

    # ── YAMNet ──────────────────────────────────────────────────────────────
    if yamnet_bundle is not None:
        try:
            import torch
            model = yamnet_bundle["model"]
            to_input = yamnet_bundle["to_input"]
            # WaveformToInput expects shape (num_channels, num_samples) and
            # the sample rate. It produces batched mel patches of shape
            # [N, C, T] ready to feed into YAMNet.
            wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # (1, samples)
            if next(model.parameters()).is_cuda:
                wav = wav.cuda()
            with torch.no_grad():
                patches = to_input(wav, sr)              # (N, C, T)
                logits = model(patches, to_prob=False)   # (N, 521) raw logits
                # YAMNet is multi-label, so use sigmoid rather than softmax.
                scores = torch.sigmoid(logits).cpu().numpy()

            music_idx = np.array(config.YAMNET_MUSIC_CLASSES, dtype=int)
            speech_idx = np.array(config.YAMNET_SPEECH_CLASSES, dtype=int)

            per_frame_music = scores[:, music_idx].sum(axis=1)   # sum of music + instrument
            per_frame_speech = scores[:, speech_idx].mean(axis=1)

            music_presence = float(per_frame_music.mean())
            music_fraction = float(
                (per_frame_music > config.MUSIC_FRAME_THRESHOLD).mean()
            )
            # Cap at 100 to prevent the ratio from exploding when music is
            # near-zero (pure-speech segments). A 0.01 floor in the denominator
            # plus a 100 ceiling keeps the feature well-conditioned for
            # downstream z-score normalisation.
            raw_ratio = per_frame_speech.mean() / max(music_presence, 0.01)
            speech_music_ratio = float(min(raw_ratio, 100.0))
            out["music_presence"] = round(music_presence, 6)
            out["music_fraction"] = round(music_fraction, 6)
            out["speech_music_ratio"] = round(speech_music_ratio, 6)
        except Exception as e:
            log.warning("yamnet infer failed for %s: %s", job.segment_id, e)

    # ── Gillick laughter ────────────────────────────────────────────────────
    if gillick_bundle is not None:
        try:
            out["laughter_peak"] = round(_gillick_peak(y, sr, gillick_bundle), 6)
        except Exception as e:
            log.debug("gillick infer failed for %s: %s", job.segment_id, e)

    return out


_GILLICK_N_MELS = 128  # ResNetBigger was trained on librosa default 128 mels


def _featurize_melspec_gillick(y: np.ndarray, sr: int = 8000, hop_length: int = 186) -> np.ndarray:
    """Gillick-compatible log-mel spectrogram.

    Replicates jrgillick/laughter-detection/utils/audio_utils.featurize_melspec
    but uses modern librosa's keyword-only API (0.10+). The original repo
    function uses positional args which fail on librosa >=0.9.

    n_mels is pinned to 128 to match the trained ResNetBigger weights.
    Output shape: (n_time_frames, 128).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=_GILLICK_N_MELS,
    ).T
    S = librosa.amplitude_to_db(S, ref=np.max)
    # Defensive: shape contract with the trained model.
    assert S.shape[1] == _GILLICK_N_MELS, (
        f"Gillick melspec n_mels mismatch: got {S.shape[1]}, expected {_GILLICK_N_MELS}"
    )
    return S


def _gillick_peak(y: np.ndarray, sr: int, bundle: dict) -> float:
    """Run Gillick laughter detector and return peak window probability.

    Mirrors jrgillick's segment_laughter.py exactly:
      1. Resample audio to 8 kHz.
      2. log-mel spectrogram, hop_length=186, default n_mels=128, amp->dB.
      3. Build sliding windows of length n_frames=44 (~1 s at hop 186/8000).
      4. Add channel dim via expand_channel_dim -> (batch, 1, 44, 128).
      5. Forward through ResNetBigger; outputs sigmoid laughter prob per window.
      6. Return max over all windows in the 30 s segment.
    """
    import torch
    model = bundle["model"]
    n_frames = 44  # hardcoded in SwitchBoardLaughterInferenceDataset

    device = next(model.parameters()).device

    # Gillick expects 8 kHz mono.
    if sr != 8000:
        y8 = librosa.resample(y, orig_sr=sr, target_sr=8000)
    else:
        y8 = y

    try:
        features = _featurize_melspec_gillick(y8.astype(np.float32), sr=8000, hop_length=186)
    except Exception as e:
        log.debug("gillick featurize failed: %s", e)
        return 0.0

    if features is None or len(features) < n_frames:
        return 0.0

    # Build all sliding windows in one batched tensor.
    # features shape: (T, 40). We want (T - n_frames, n_frames, 40).
    T = features.shape[0]
    num_windows = T - n_frames + 1
    if num_windows <= 0:
        return 0.0

    # Strided slicing, then add channel dim: (num_windows, 1, n_frames, 40)
    windows = np.stack(
        [features[i : i + n_frames] for i in range(num_windows)],
        axis=0,
    )
    windows = np.expand_dims(windows, 1).astype(np.float32)

    # Batch-forward in chunks to avoid VRAM spikes on long clips.
    batch_size = 128
    peak = 0.0
    with torch.no_grad():
        for start in range(0, num_windows, batch_size):
            chunk = torch.from_numpy(windows[start : start + batch_size]).to(device)
            preds = model(chunk)
            # Model already applies sigmoid in forward. Range [0, 1].
            preds_np = preds.detach().cpu().numpy().squeeze(-1)
            chunk_max = float(preds_np.max())
            if chunk_max > peak:
                peak = chunk_max
    return peak
