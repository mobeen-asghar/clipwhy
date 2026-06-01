"""Creator context (1 logical, 5 physical columns via one-hot on category).

Size is deliberately NOT a feature here: the label is creator-relative
(median-based), so subscriber count and creator median are absorbed by the
target and adding them as features is redundant and leakage-adjacent.
"""
from .. import config
from ..pipeline import SegmentJob


def extract(job: SegmentJob) -> dict:
    out = {f"creator_category_{c}": 0.0 for c in config.CATEGORY_ORDER}
    key = f"creator_category_{job.category}"
    if key in out:
        out[key] = 1.0
    return out
