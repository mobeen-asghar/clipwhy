"""Audio speech features (7, librosa).

Formulas match V1 exactly (V1-parity for MSc ablation), minus spectral_centroid.
"""
import logging

import librosa
import numpy as np

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.audio_speech")


_ZERO = {
    "energy_mean": 0.0, "energy_var": 0.0, "energy_first_3s_ratio": 1.0,
    "pitch_range": 0.0, "pitch_std": 0.0,
    "speaking_rate_audio": 0.0, "silence_ratio": 0.0,
}


def extract(job: SegmentJob) -> dict:
    try:
        y, sr = librosa.load(str(job.audio_path), sr=config.AUDIO_SAMPLE_RATE)
    except Exception as e:
        log.warning("librosa load failed for %s: %s", job.segment_id, e)
        return dict(_ZERO)

    if len(y) == 0:
        return dict(_ZERO)

    duration_sec = len(y) / sr

    # ── Energy ──────────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_var = float(np.var(rms))

    hook_samples = sr * config.HOOK_WINDOW_SEC
    if len(y) > hook_samples:
        rms_first = librosa.feature.rms(y=y[:hook_samples])[0]
        mean_first = float(np.mean(rms_first))
        energy_first_3s_ratio = float(mean_first / (energy_mean + 1e-8))
    else:
        energy_first_3s_ratio = 1.0

    # ── Pitch via pyin ──────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=config.PYIN_FMIN, fmax=config.PYIN_FMAX, sr=sr,
        )
        voiced_mask = ~np.isnan(f0)
        voiced_frac = float(np.sum(voiced_mask) / max(len(f0), 1))
        if voiced_frac >= config.MIN_VOICED_FRACTION and np.sum(voiced_mask) >= 5:
            f0_clean = f0[voiced_mask]
            pitch_range = float(np.max(f0_clean) - np.min(f0_clean))
            pitch_std = float(np.std(f0_clean))
        else:
            pitch_range = 0.0
            pitch_std = 0.0
    except Exception as e:
        log.debug("pyin failed for %s: %s", job.segment_id, e)
        pitch_range = 0.0
        pitch_std = 0.0

    # ── Speaking rate (onset density) ───────────────────────────────────────
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=5,
        )
        speaking_rate_audio = float(len(peaks) / duration_sec) if duration_sec > 0 else 0.0
    except Exception:
        speaking_rate_audio = 0.0

    # ── Silence ratio (adaptive threshold) ──────────────────────────────────
    if energy_mean > config.SILENCE_FALLBACK:
        threshold = energy_mean * config.SILENCE_ADAPTIVE_RATIO
    else:
        threshold = config.SILENCE_FALLBACK
    silence_ratio = float(np.sum(rms < threshold) / max(len(rms), 1))

    return {
        "energy_mean": round(energy_mean, 6),
        "energy_var": round(energy_var, 8),
        "energy_first_3s_ratio": round(energy_first_3s_ratio, 4),
        "pitch_range": round(pitch_range, 2),
        "pitch_std": round(pitch_std, 2),
        "speaking_rate_audio": round(speaking_rate_audio, 4),
        "silence_ratio": round(silence_ratio, 4),
    }
