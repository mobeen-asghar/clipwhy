"""Voice quality features (2, parselmouth / Praat).

jitter_local and shimmer_local from the canonical Praat recipe.
Gated by the same voicing threshold as V1 pitch features, raised to 20%
because Praat needs more voiced periods to compute stable jitter/shimmer.

Per the arXiv 2602.22299 'Decoding the Hook' paper and Teixeira 2013's
clinical review, the "local" variants are the canonical first choice; we
deliberately skip jitter_rap, ppq5, apq3, apq5, apq11, HNR to avoid
multicollinearity (all >0.9 correlated with the local variants in normal speech).
"""
import logging

import numpy as np

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.voice_quality")


_ZERO = {"jitter_local": 0.0, "shimmer_local": 0.0}


def extract(job: SegmentJob) -> dict:
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        log.warning("parselmouth not installed; voice_quality features will be zero")
        return dict(_ZERO)

    try:
        snd = parselmouth.Sound(str(job.audio_path))
    except Exception as e:
        log.warning("parselmouth load failed for %s: %s", job.segment_id, e)
        return dict(_ZERO)

    # Voicing gate: if too little voiced speech in the segment, skip.
    try:
        pitch = call(snd, "To Pitch", 0.0, config.PYIN_FMIN, config.PYIN_FMAX)
        voiced_frac = float(
            call(pitch, "Count voiced frames") / max(call(pitch, "Get number of frames"), 1)
        )
    except Exception:
        voiced_frac = 0.0
    if voiced_frac < config.MIN_VOICED_FRAC_FOR_PRAAT:
        return dict(_ZERO)

    try:
        pp = call(snd, "To PointProcess (periodic, cc)", config.PYIN_FMIN, config.PYIN_FMAX)
        jitter_local = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer_local = float(
            call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )
    except Exception as e:
        log.debug("praat jitter/shimmer failed for %s: %s", job.segment_id, e)
        return dict(_ZERO)

    # Praat can emit NaN/undefined on sparse periods; clamp to zero.
    if not np.isfinite(jitter_local):
        jitter_local = 0.0
    if not np.isfinite(shimmer_local):
        shimmer_local = 0.0

    return {
        "jitter_local": round(jitter_local, 6),
        "shimmer_local": round(shimmer_local, 6),
    }
