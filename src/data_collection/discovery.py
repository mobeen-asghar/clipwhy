"""Stage 1: Candidate generation from multiple sources.

Sources (by effectiveness):
  1. Seeds: 16.6% conversion, highest quality (known good channels)
  2. Video search (type=video): 2.6% conversion, cheapest per creator
  3. Channel search (type=channel): 1.0% conversion, highest volume

Removed:
  - channelSections crawl: YouTube removed featured channels UI in Nov 2023.
    Produced 0 results in full run. Dead feature.
"""

import json
import logging

import pandas as pd

log = logging.getLogger("clipwhy.discovery")


def _load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ── 1a: Resolve V1 seed channel names to IDs ─────────────────────────────────

def resolve_seeds(api, seeds_dict, cache_dir):
    """Search YouTube for each seed name, return {category: [channel_ids]}."""
    cache_path = cache_dir / "seed_resolved.json"
    cache = _load_json(cache_path)

    result = {}
    for category, names in seeds_dict.items():
        ids = []
        for name in names:
            if name in cache:
                if cache[name]:
                    ids.append(cache[name])
                continue
            items = api.search(name, search_type="channel", max_results=1)
            if items:
                cid = items[0]["snippet"]["channelId"]
                cache[name] = cid
                ids.append(cid)
                log.info("  Seed [%s] %s -> %s", category, name, cid)
            else:
                cache[name] = None
                log.warning("  Seed [%s] %s -> NOT FOUND", category, name)
            _save_json(cache_path, cache)
        result[category] = ids
        log.info("  [%s] %d seeds resolved", category, len(ids))

    total = sum(len(v) for v in result.values())
    log.info("Total seeds resolved: %d", total)
    return result


# ── 1b: Search for channels (type=channel) ───────────────────────────────────

def search_channels(api, queries_dict, topic_ids):
    """Search with type=channel + topicId per category."""
    result = {}
    for category, queries in queries_dict.items():
        seen = set()
        topic_id = topic_ids.get(category)

        for i, q in enumerate(queries):
            try:
                # With topic ID
                if topic_id:
                    items = api.search(q, search_type="channel", topic_id=topic_id)
                    for item in items:
                        seen.add(item["snippet"]["channelId"])

                # Without topic ID (broader, different results)
                items = api.search(q, search_type="channel")
                for item in items:
                    seen.add(item["snippet"]["channelId"])

            except RuntimeError:
                log.error("API exhausted during channel search [%s]", category)
                break
            except Exception as e:
                log.warning("Channel search '%s': %s", q, e)

            if (i + 1) % 10 == 0:
                log.info("  [%s] %d/%d queries, %d unique channels",
                         category, i + 1, len(queries), len(seen))

        result[category] = list(seen)
        log.info("[%s] Channel search: %d unique channels", category, len(seen))

    return result


# ── 1c: Search for videos (type=video), extract channel IDs ──────────────────

def search_videos(api, queries_dict):
    """Search with type=video, extract unique channel IDs."""
    result = {}
    for category, queries in queries_dict.items():
        seen = set()

        for i, q in enumerate(queries):
            try:
                items = api.search(q, search_type="video")
                for item in items:
                    seen.add(item["snippet"]["channelId"])
            except RuntimeError:
                log.error("API exhausted during video search [%s]", category)
                break
            except Exception as e:
                log.warning("Video search '%s': %s", q, e)

            if (i + 1) % 10 == 0:
                log.info("  [%s] %d/%d queries, %d unique channels",
                         category, i + 1, len(queries), len(seen))

        result[category] = list(seen)
        log.info("[%s] Video search: %d unique channels", category, len(seen))

    return result


# ── Merge all sources ─────────────────────────────────────────────────────────

def generate_candidates(api, seeds_dict, queries_dict, topic_ids,
                        cache_dir, viral_queries=None, test_mode=False):
    """
    Run all discovery sources and merge into a deduplicated candidates DataFrame.
    Columns: channel_id, source, search_category
    """
    if test_mode:
        seeds_dict = {k: v[:5] for k, v in seeds_dict.items()}
        queries_dict = {k: v[:8] for k, v in queries_dict.items()}
        viral_queries = (viral_queries or [])[:3]
    else:
        viral_queries = viral_queries or []

    # Append viral queries to each category
    if viral_queries:
        queries_dict = {
            cat: queries + viral_queries
            for cat, queries in queries_dict.items()
        }

    # 1a: Seeds
    log.info("=" * 60)
    log.info("Stage 1a: Resolving seed channels")
    log.info("=" * 60)
    seed_map = resolve_seeds(api, seeds_dict, cache_dir)

    # 1b: Channel search
    log.info("=" * 60)
    log.info("Stage 1b: Searching for channels (type=channel)")
    log.info("=" * 60)
    ch_search = search_channels(api, queries_dict, topic_ids)

    # 1c: Video search
    log.info("=" * 60)
    log.info("Stage 1c: Searching for videos (type=video)")
    log.info("=" * 60)
    vid_search = search_videos(api, queries_dict)

    # Merge and deduplicate (first occurrence determines category)
    seen = set()
    rows = []

    def add(channel_id, source, category):
        if channel_id not in seen:
            seen.add(channel_id)
            rows.append({
                "channel_id": channel_id,
                "source": source,
                "search_category": category,
            })

    # Seeds first (highest conversion: 16.6%)
    for cat, ids in seed_map.items():
        for cid in ids:
            add(cid, "seed", cat)

    # Channel search (1.0% conversion but highest volume)
    for cat, ids in ch_search.items():
        for cid in ids:
            add(cid, "channel_search", cat)

    # Video search (2.6% conversion, cheapest per creator)
    for cat, ids in vid_search.items():
        for cid in ids:
            add(cid, "video_search", cat)

    df = pd.DataFrame(rows)
    log.info("Stage 1 total: %d unique candidates", len(df))
    for src in ["seed", "channel_search", "video_search"]:
        n = len(df[df["source"] == src])
        if n:
            log.info("  %s: %d", src, n)

    return df
