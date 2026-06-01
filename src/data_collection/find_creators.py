"""
ClipWhy V2 Creator Discovery Pipeline.

5-stage pipeline to find YouTube creators who post both Shorts and
long-form videos on the SAME channel and repurpose content between them.

  Stage 1: Candidate generation (seeds, channel sections crawl, search)
  Stage 2: Batch channel filter (subs, country, category)
  Stage 3: Video count verification (Shorts + longs on same channel)
  Stage 4: English verification (Whisper)
  Stage 5: Repurpose verification (caption fuzzy matching, no captions = skip)

Usage:
    ./run.sh                      # full run
    ./run.sh --test               # small test run
    ./run.sh --stage 3            # resume from Stage 3
    ./run.sh --test --stage 2     # test from Stage 2
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from config.settings import (
    YOUTUBE_API_KEYS, DISCORD_WEBHOOK_URL, DATA_DIR, CACHE_DIR,
    CATEGORIES, TARGET_TOTAL,
    V1_SEEDS, SEARCH_QUERIES, TOPIC_IDS,
    SIZE_BRACKETS, VIRAL_QUERIES,
)
from src.data_collection.youtube_api import YouTubeAPI
from src.data_collection.notifier import Notifier
from src.data_collection.discovery import generate_candidates
from src.data_collection.filters import batch_channel_filter, video_count_filter
from src.data_collection.verify import verify_english, verify_repurpose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clipwhy.main")

# Stage output files
STAGE1_FILE = DATA_DIR / "stage1_candidates.csv"
STAGE2_FILE = DATA_DIR / "stage2_filtered.csv"
STAGE3_FILE = DATA_DIR / "stage3_counts.csv"
STAGE4_FILE = DATA_DIR / "stage4_english.csv"
STAGE5_FILE = DATA_DIR / "stage5_repurpose.csv"
CREATORS_FILE = DATA_DIR / "creators.csv"


def _get_bracket(subs):
    for name, (lo, hi) in SIZE_BRACKETS.items():
        if lo <= subs < hi:
            return name
    return "large"


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage1(api, notify, test_mode):
    if STAGE1_FILE.exists():
        df = pd.read_csv(STAGE1_FILE)
        notify.send(f"Stage 1: Loaded {len(df)} candidates from cache")
        return df

    notify.send("Stage 1: Candidate generation...")
    df = generate_candidates(
        api, V1_SEEDS, SEARCH_QUERIES, TOPIC_IDS,
        CACHE_DIR,
        viral_queries=VIRAL_QUERIES, test_mode=test_mode,
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGE1_FILE, index=False)

    notify.send(f"Stage 1 complete: {len(df)} candidates")
    for cat in CATEGORIES:
        n = len(df[df["search_category"] == cat]) if len(df) else 0
        notify.send(f"  {cat}: {n}")
    return df


def run_stage2(api, candidates, notify):
    if STAGE2_FILE.exists():
        df = pd.read_csv(STAGE2_FILE)
        notify.send(f"Stage 2: Loaded {len(df)} filtered from cache")
        return df

    notify.send(f"Stage 2: Batch filtering {len(candidates)} channels...")
    df = batch_channel_filter(api, candidates)
    df.to_csv(STAGE2_FILE, index=False)

    notify.send(f"Stage 2 complete: {len(df)} passed")
    for cat in CATEGORIES:
        n = len(df[df["category"] == cat]) if len(df) else 0
        notify.send(f"  {cat}: {n}")
    return df


def run_stage3(api, filtered, notify):
    if STAGE3_FILE.exists():
        df = pd.read_csv(STAGE3_FILE)
        notify.send(f"Stage 3: Loaded {len(df)} verified from cache")
        return df

    notify.send(f"Stage 3: Video count check for {len(filtered)} channels...")
    df = video_count_filter(api, filtered, CACHE_DIR, notify=notify)
    df.to_csv(STAGE3_FILE, index=False)

    notify.send(f"Stage 3 complete: {len(df)} have Shorts + longs on same channel")
    for cat in CATEGORIES:
        n = len(df[df["category"] == cat]) if len(df) else 0
        notify.send(f"  {cat}: {n}")
    return df


def run_stage4(verified, notify):
    if STAGE4_FILE.exists():
        df = pd.read_csv(STAGE4_FILE)
        notify.send(f"Stage 4: Loaded {len(df)} English from cache")
        return df

    notify.send(f"Stage 4: English verification for {len(verified)} channels...")
    df = verify_english(verified, CACHE_DIR, notify=notify)
    df.to_csv(STAGE4_FILE, index=False)

    notify.send(f"Stage 4 complete: {len(df)} English verified")
    for cat in CATEGORIES:
        n = len(df[df["category"] == cat]) if len(df) else 0
        notify.send(f"  {cat}: {n}")
    return df


def run_stage5(english, notify):
    if STAGE5_FILE.exists():
        df = pd.read_csv(STAGE5_FILE)
        notify.send(f"Stage 5: Loaded {len(df)} repurposers from cache")
        return df

    notify.send(f"Stage 5: Repurpose verification for {len(english)} channels...")
    df = verify_repurpose(english, CACHE_DIR, notify=notify)
    df.to_csv(STAGE5_FILE, index=False)

    notify.send(f"Stage 5 complete: {len(df)} confirmed repurposers")
    for cat in CATEGORIES:
        n = len(df[df["category"] == cat]) if len(df) else 0
        notify.send(f"  {cat}: {n}")
    return df


# ── Final selection ───────────────────────────────────────────────────────────

def select_creators(repurposed_df, notify):
    """Keep all verified creators, no cap. Sort by subscribers per category."""
    if repurposed_df.empty:
        notify.error("No repurposed creators to select from")
        return pd.DataFrame()

    selected = []
    for cat in CATEGORIES:
        cat_df = repurposed_df[repurposed_df["category"] == cat]
        cat_df = cat_df.sort_values("subscribers", ascending=False)
        selected.append(cat_df)
        notify.send(f"  {cat}: {len(cat_df)} creators")

    df = pd.concat(selected, ignore_index=True)

    # No cap: keep ALL verified creators. More data = better models.
    # TARGET_TOTAL is a minimum goal, not a maximum.

    # Assign creator IDs and clean up
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        subs = int(row.get("subscribers", 0))
        rows.append({
            "creator_id": f"CR{i:04d}",
            "channel_id": row["channel_id"],
            "channel_title": row.get("channel_title", ""),
            "channel_url": f"https://youtube.com/channel/{row['channel_id']}",
            "subscribers": subs,
            "subscriber_bracket": _get_bracket(subs),
            "category": row["category"],
            "country": row.get("country", ""),
            "shorts_count": int(row.get("shorts_count", 0)),
            "longs_count": int(row.get("longs_count", 0)),
            "comments_disabled": bool(row.get("comments_disabled", False)),
            "whisper_en_prob": float(row.get("whisper_en_prob", 0)),
            "repurpose_matches": int(row.get("repurpose_matches", 0)),
            "collection_date": now,
        })

    result = pd.DataFrame(rows)
    result.to_csv(CREATORS_FILE, index=False)

    notify.send(f"\nFinal: {len(result)} creators saved to {CREATORS_FILE}")
    for cat in CATEGORIES:
        n = len(result[result["category"] == cat])
        notify.send(f"  {cat}: {n}")
    for bracket in SIZE_BRACKETS:
        n = len(result[result["subscriber_bracket"] == bracket])
        notify.send(f"  {bracket}: {n}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClipWhy V2 Creator Discovery")
    parser.add_argument("--stage", type=int, default=1, help="Resume from stage (1-5)")
    parser.add_argument("--test", action="store_true", help="Small test run")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    notify = Notifier(
        webhook_url=DISCORD_WEBHOOK_URL,
        log_path=DATA_DIR / "discovery_log.txt",
    )
    api = YouTubeAPI(YOUTUBE_API_KEYS, notify=notify)

    notify.send("=" * 60)
    notify.send("ClipWhy V2 Creator Discovery")
    notify.send("=" * 60)
    notify.send(f"  API keys: {len(YOUTUBE_API_KEYS)}")
    notify.send(f"  Target: {TARGET_TOTAL}+ creators (no cap, keeping all verified)")
    notify.send(f"  Mode: {'TEST' if args.test else 'FULL'}")
    notify.send(f"  Start stage: {args.stage}")

    try:
        # Helper: load prior stage CSV or fail with clear message
        def _load_stage(path, stage_num):
            if not path.exists():
                notify.error(f"Cannot resume from stage {args.stage}: "
                             f"{path.name} not found. Run earlier stages first.")
                sys.exit(1)
            return pd.read_csv(path)

        # Stage 1: Candidate generation
        if args.stage <= 1:
            candidates = run_stage1(api, notify, args.test)
        else:
            candidates = _load_stage(STAGE1_FILE, 1)

        # Stage 2: Batch channel filter
        if args.stage <= 2:
            filtered = run_stage2(api, candidates, notify)
        else:
            filtered = _load_stage(STAGE2_FILE, 2)

        # Stage 3: Video count verification
        if args.stage <= 3:
            verified = run_stage3(api, filtered, notify)
        else:
            verified = _load_stage(STAGE3_FILE, 3)

        # Stage 4: English verification
        if args.stage <= 4:
            english = run_stage4(verified, notify)
        else:
            english = _load_stage(STAGE4_FILE, 4)

        # Stage 5: Repurpose verification
        if args.stage <= 5:
            repurposed = run_stage5(english, notify)
        else:
            repurposed = _load_stage(STAGE5_FILE, 5)

        # Final selection
        notify.send("=" * 60)
        notify.send("Selecting final creators...")
        notify.send("=" * 60)
        creators = select_creators(repurposed, notify)

        # Summary
        notify.send(f"\n{'=' * 60}")
        notify.send("DISCOVERY COMPLETE")
        notify.send(f"{'=' * 60}")
        notify.send(f"  Stage 1 (candidates):  {len(candidates)}")
        notify.send(f"  Stage 2 (filtered):    {len(filtered)}")
        notify.send(f"  Stage 3 (counts):      {len(verified)}")
        notify.send(f"  Stage 4 (English):     {len(english)}")
        notify.send(f"  Stage 5 (repurpose):   {len(repurposed)}")
        notify.send(f"  Final creators:        {len(creators)}")

        notify.send("\n  API usage:")
        for key, usage in api.get_quota_status().items():
            if usage["used"] > 0:
                notify.send(f"    {key}: {usage['used']} units"
                            f"{' (EXHAUSTED)' if usage['exhausted'] else ''}")

        notify.done(f"{len(creators)} creators found")

    except Exception as e:
        notify.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
