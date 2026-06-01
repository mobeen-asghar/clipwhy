"""Stages 2-3: Batch channel filtering and video count verification."""

import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from config.settings import (
    MIN_SUBSCRIBERS, MIN_LONG_VIDEOS, MIN_SHORTS,
    MIN_LONG_VIDEO_DURATION_SEC, MAX_SHORT_DURATION_SEC,
    ACTIVE_WITHIN_DAYS, ENGLISH_COUNTRIES, NON_ENGLISH_COUNTRIES,
    CATEGORIES,
)
from src.data_collection.youtube_api import YouTubeAPI

log = logging.getLogger("clipwhy.filters")


def _load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# Weighted topic scoring (from YouTube API topicCategories)
# Sports -> entertainment (not fitness). Knowledge is weak signal.
TOPIC_WEIGHTS = {
    "Technology": ("tech", 5), "Computer_programming": ("tech", 5),
    "Software": ("tech", 5), "Computer_hardware": ("tech", 4), "Gadget": ("tech", 3),
    "Education": ("education", 5), "Mathematics": ("education", 5),
    "Philosophy": ("education", 4), "Psychology": ("education", 4),
    "Science": ("education", 4), "Knowledge": ("education", 2),
    "Physical_fitness": ("fitness", 5), "Physical_exercise": ("fitness", 5),
    "Nutrition": ("fitness", 5), "Health": ("fitness", 4), "Medicine": ("fitness", 4),
    "Politics": ("commentary", 5), "Military": ("commentary", 4),
    "Business": ("commentary", 3), "Society": ("commentary", 2),
    "Entertainment": ("entertainment", 3), "Music": ("entertainment", 4),
    "Film": ("entertainment", 3), "Television_program": ("entertainment", 4),
    "Performing_arts": ("entertainment", 4), "Humor": ("entertainment", 5),
    "Comedy": ("entertainment", 5), "Video_game": ("entertainment", 4),
    "Sport": ("entertainment", 3), "American_football": ("entertainment", 3),
    "Association_football": ("entertainment", 3), "Basketball": ("entertainment", 3),
    "Mixed_martial_arts": ("entertainment", 3), "Motorsport": ("entertainment", 3),
    "Lifestyle_(sociology)": ("entertainment", 1), "Food": ("entertainment", 2),
    "Pet": ("entertainment", 2), "Tourism": ("entertainment", 2),
    "Fashion": ("entertainment", 2),
}


def _category_from_topics(topic_details):
    """Weighted scoring from YouTube topicDetails.topicCategories."""
    cats = topic_details.get("topicCategories", [])
    scores = {"tech": 0, "education": 0, "entertainment": 0, "fitness": 0, "commentary": 0}
    for url in cats:
        topic = url.rsplit("/", 1)[-1] if "/" in url else ""
        if topic in TOPIC_WEIGHTS:
            cat, weight = TOPIC_WEIGHTS[topic]
            scores[cat] += weight
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _passes_country_filter(country, default_language):
    """Pre-filter obvious non-English channels by country + language tag."""
    if not country:
        return True  # unknown country, let Whisper decide
    country = country.upper()
    if country in ENGLISH_COUNTRIES:
        return True
    if default_language and default_language.lower().startswith("en"):
        return True
    if country in NON_ENGLISH_COUNTRIES:
        return False
    return True  # unlisted country, pass through


# ── Stage 2: Batch channel filter ─────────────────────────────────────────────

def batch_channel_filter(api, candidates_df):
    """
    Batch-fetch channel details. Filter by subs, total videos, country.
    Assign category from topicDetails (with search_category fallback).
    """
    all_ids = candidates_df["channel_id"].tolist()
    cat_map = dict(zip(candidates_df["channel_id"], candidates_df["search_category"]))

    log.info("Stage 2: Fetching details for %d channels", len(all_ids))

    passed = []
    total_fetched = 0

    for i in range(0, len(all_ids), 50):
        batch = all_ids[i:i + 50]
        try:
            channels = api.get_channel_details(batch)
        except RuntimeError:
            log.error("API exhausted in Stage 2")
            break

        for ch in channels:
            cid = ch["id"]
            stats = ch.get("statistics", {})
            snippet = ch.get("snippet", {})
            content = ch.get("contentDetails", {})
            topics = ch.get("topicDetails", {})

            subs = int(stats.get("subscriberCount", 0))
            total_vids = int(stats.get("videoCount", 0))

            if subs < MIN_SUBSCRIBERS:
                continue
            if total_vids < (MIN_LONG_VIDEOS + MIN_SHORTS):
                continue

            country = snippet.get("country", "")
            default_lang = snippet.get("defaultLanguage", "")

            if not _passes_country_filter(country, default_lang):
                continue

            # Category: prefer YouTube's topicDetails (weighted scoring),
            # fall back to search query category if YouTube has no topic data
            topic_cat = _category_from_topics(topics)
            search_cat = cat_map.get(cid, "unknown")
            category = topic_cat or (search_cat if search_cat in CATEGORIES else "entertainment")

            passed.append({
                "channel_id": cid,
                "channel_title": snippet.get("title", ""),
                "subscribers": subs,
                "total_videos": total_vids,
                "country": country,
                "default_language": default_lang,
                "category": category,
                "uploads_playlist_id": content.get("relatedPlaylists", {}).get("uploads", ""),
            })

        total_fetched += len(batch)
        if total_fetched % 500 == 0:
            log.info("  Fetched %d/%d, %d passed so far",
                     total_fetched, len(all_ids), len(passed))

    log.info("Stage 2 complete: %d/%d passed", len(passed), len(all_ids))
    return pd.DataFrame(passed)


# ── Stage 3: Video count verification ─────────────────────────────────────────

def video_count_filter(api, filtered_df, cache_dir, notify=None):
    """
    For each channel, fetch recent 200 videos and count Shorts vs longs.
    Both must be on the SAME channel.
    Saves video IDs to cache for Stage 5.
    """
    video_ids_cache_path = cache_dir / "video_ids.json"
    video_ids_cache = _load_json(video_ids_cache_path)

    progress_cache_path = cache_dir / "stage3_progress.json"
    progress = _load_json(progress_cache_path)

    passed = []
    checked = 0

    for _, row in filtered_df.iterrows():
        cid = row["channel_id"]
        playlist = row.get("uploads_playlist_id", "")

        if not playlist:
            continue

        # Check progress cache
        if cid in progress:
            if progress[cid].get("passed"):
                result = dict(progress[cid])
                result.update(row.to_dict())
                result.pop("passed", None)
                result.pop("reason", None)
                passed.append(result)
            checked += 1
            if checked % 50 == 0:
                msg = (f"  Stage 3: {checked}/{len(filtered_df)} checked, "
                       f"{len(passed)} passed")
                log.info(msg)
                if notify:
                    notify.send(msg)
            continue

        try:
            vids = api.list_playlist_videos(playlist, max_videos=200)
            if not vids:
                progress[cid] = {"passed": False, "reason": "no_videos"}
                continue

            details = api.get_video_details(vids)

            short_ids = []
            long_ids = []
            latest = ""

            for v in details:
                dur = YouTubeAPI.parse_duration(
                    v.get("contentDetails", {}).get("duration", "")
                )
                vid = v["id"]
                pub = v.get("snippet", {}).get("publishedAt", "")

                if 0 < dur <= MAX_SHORT_DURATION_SEC:
                    short_ids.append(vid)
                elif dur >= MIN_LONG_VIDEO_DURATION_SEC:
                    long_ids.append(vid)

                if pub > latest:
                    latest = pub

            if len(short_ids) < MIN_SHORTS or len(long_ids) < MIN_LONG_VIDEOS:
                progress[cid] = {
                    "passed": False,
                    "reason": f"shorts={len(short_ids)},longs={len(long_ids)}",
                }
                checked += 1
                continue

            # Activity check
            if latest:
                try:
                    lt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                    cutoff = datetime.now(lt.tzinfo) - timedelta(days=ACTIVE_WITHIN_DAYS)
                    if lt < cutoff:
                        progress[cid] = {"passed": False, "reason": "inactive"}
                        checked += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # Comments disabled check (sample first 20 Shorts)
            sample_short_ids = set(short_ids[:20])
            shorts_details = [v for v in details
                              if v["id"] in sample_short_ids]
            comments_on = sum(
                1 for v in shorts_details
                if int(v.get("statistics", {}).get("commentCount", 0)) > 0
            )
            comments_disabled = (comments_on == 0) if shorts_details else False

            # Save video IDs for Stages 4-5
            short5 = short_ids[:5]
            long5 = long_ids[:5]
            video_ids_cache[cid] = {
                "short_ids": short5,
                "long_ids": long5,
            }

            result = {
                "passed": True,
                "shorts_count": len(short_ids),
                "longs_count": len(long_ids),
                "latest_video_date": latest,
                "comments_disabled": comments_disabled,
                # Store video IDs in CSV so they survive VM migrations
                "short_video_ids": ",".join(short5),
                "long_video_ids": ",".join(long5),
            }
            progress[cid] = result

            row_dict = row.to_dict()
            row_dict.update(result)
            del row_dict["passed"]
            passed.append(row_dict)

        except RuntimeError:
            log.error("API exhausted in Stage 3")
            break
        except Exception as e:
            log.warning("Stage 3 error %s: %s", cid, e)
            progress[cid] = {"passed": False, "reason": str(e)[:50]}

        checked += 1
        if checked % 50 == 0:
            _save_json(progress_cache_path, progress)
            _save_json(video_ids_cache_path, video_ids_cache)
            msg = (f"  Stage 3: {checked}/{len(filtered_df)} checked, "
                   f"{len(passed)} passed")
            log.info(msg)
            if notify:
                notify.send(msg)

    _save_json(progress_cache_path, progress)
    _save_json(video_ids_cache_path, video_ids_cache)

    log.info("Stage 3 complete: %d/%d passed", len(passed), len(filtered_df))
    return pd.DataFrame(passed)
