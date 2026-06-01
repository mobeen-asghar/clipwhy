"""
Engagement metadata collection, baseline computation, and virality thresholds.

Called in Step 5 (after Whisper matching, not before).

Adapted from V1:
  - clipwhy-scraper/collect_engagement.py (metadata fetch, ER/VPD)
  - clipwhy-scraper/compute_labels.py (baselines, virality auto-adjustment)
"""

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from . import config

log = logging.getLogger("clipwhy.engagement")


def safe_divide(a, b, default=0.0):
    return a / b if b > 0 else default


def fetch_shorts_metadata(
    api, short_ids: list[str], creator_id: str, comments_disabled: bool
) -> pd.DataFrame:
    """Fetch views/likes/comments for a list of shorts.
    Returns DataFrame with engagement columns."""
    now = datetime.now(timezone.utc)
    min_age_cutoff = now - timedelta(days=config.MIN_SHORT_AGE_DAYS)
    collected_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Batch fetch video details and normalize from raw API format
    raw_items = api.get_video_details(short_ids)
    videos = [config.normalize_video_item(item) for item in raw_items]

    rows = []
    for v in videos:
        if not v["published_at"]:
            continue
        pub = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
        days_live = max((now - pub).days, 1)
        too_recent = pub > min_age_cutoff

        views = v["views"]
        likes = v["likes"]
        comments = v["comments"]

        if comments_disabled:
            engagement_rate = safe_divide(likes, views)
            formula = "likes/views"
        else:
            engagement_rate = safe_divide(likes + comments, views)
            formula = "standard"

        views_per_day = safe_divide(views, days_live)

        rows.append({
            "video_id": v["video_id"],
            "creator_id": creator_id,
            "title": v["title"],
            "views": views,
            "likes": likes,
            "comments": comments,
            "published_at": v["published_at"],
            "duration_seconds": v["duration_seconds"],
            "days_since_published": days_live,
            "views_per_day": round(views_per_day, 1),
            "engagement_rate": round(engagement_rate, 6),
            "engagement_formula": formula,
            "collected_at": collected_at,
            "excluded_reason": "too_recent" if too_recent else "",
            # Extra fields for future analysis
            "description": v.get("description", ""),
            "tags": v.get("tags", ""),
            "category_id": v.get("category_id", ""),
            "channel_id": v.get("channel_id", ""),
            "channel_title": v.get("channel_title", ""),
            "thumbnail_url": v.get("thumbnail_url", ""),
            "default_audio_language": v.get("default_audio_language", ""),
            "definition": v.get("definition", ""),
            "has_captions": v.get("has_captions", ""),
        })

    return pd.DataFrame(rows)


def compute_baselines(shorts_df: pd.DataFrame, creator_id: str,
                      comments_disabled: bool) -> dict:
    """Compute baselines from all eligible shorts for one creator.
    Returns dict with median/mean stats."""
    eligible = shorts_df[shorts_df["excluded_reason"] == ""]
    if len(eligible) == 0:
        log.warning("[%s] No eligible shorts for baseline", creator_id)
        return {}

    excluded_count = len(shorts_df[shorts_df["excluded_reason"] != ""])

    return {
        "creator_id": creator_id,
        "total_shorts_analysed": len(eligible),
        "shorts_excluded_too_recent": excluded_count,
        "median_engagement_rate": round(eligible["engagement_rate"].median(), 6),
        "mean_engagement_rate": round(eligible["engagement_rate"].mean(), 6),
        "median_views": int(eligible["views"].median()),
        "mean_views": int(eligible["views"].mean()),
        "std_views": int(eligible["views"].std()) if len(eligible) > 1 else 0,
        "median_views_per_day": round(eligible["views_per_day"].median(), 1),
        "mean_views_per_day": round(eligible["views_per_day"].mean(), 1),
        "median_likes": int(eligible["likes"].median()),
        "median_comments": int(eligible["comments"].median()),
        "comments_disabled": comments_disabled,
        "engagement_formula_used": "likes/views" if comments_disabled else "standard",
        "baseline_computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def label_matched_shorts(
    whisper_pairs: list[dict],
    shorts_df: pd.DataFrame,
    baselines: dict,
    eng_mult: float = None,
    vpd_mult: float = None,
) -> list[dict]:
    """Apply virality labels to Whisper-matched shorts.

    For each matched short, checks:
      - engagement_rate > eng_mult * creator_median_engagement_rate
      - views_per_day > vpd_mult * creator_median_views_per_day

    Returns list of pair dicts with label and engagement columns added.
    """
    eng_mult = eng_mult or config.ENGAGEMENT_MULTIPLIER
    vpd_mult = vpd_mult or config.VIEWS_PER_DAY_MULTIPLIER

    median_er = baselines.get("median_engagement_rate", 0)
    median_vpd = baselines.get("median_views_per_day", 0)

    # Build lookup: short_id -> engagement data
    shorts_lookup = {}
    for _, row in shorts_df.iterrows():
        shorts_lookup[row["video_id"]] = row

    labeled = []
    for pair in whisper_pairs:
        short_id = pair["short_id"]
        if short_id not in shorts_lookup:
            log.warning("Short %s not in metadata, skipping", short_id)
            continue

        s = shorts_lookup[short_id]
        er = s["engagement_rate"]
        vpd = s["views_per_day"]

        meets_er = er > eng_mult * median_er if median_er > 0 else False
        meets_vpd = vpd > vpd_mult * median_vpd if median_vpd > 0 else False
        label = 1 if (meets_er or meets_vpd) else 0

        labeled.append({
            **pair,
            "short_views": int(s["views"]),
            "short_likes": int(s["likes"]),
            "short_comments": int(s["comments"]),
            "short_engagement_rate": round(er, 6),
            "short_views_per_day": round(vpd, 1),
            "creator_median_engagement_rate": round(median_er, 6),
            "creator_median_views_per_day": round(median_vpd, 1),
            "meets_engagement_threshold": meets_er,
            "meets_views_threshold": meets_vpd,
            "label": label,
        })

    return labeled
