"""Per-category feature extractors.

Every extractor module exposes one of:
  extract(job)             -> dict of feature name -> value        (CPU, per-segment)
  extract(job, models)     -> dict                                 (GPU, per-segment)
  extract_batch(jobs, models) -> list[(dict, optional CLIP emb)]   (GPU, batched)

Model loading is done once at pod startup via worker._load_models(), with
load_* functions in the GPU-bound extractors (audio_events, audio_emotion,
visual). CPU extractors have no load step.
"""
