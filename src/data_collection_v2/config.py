"""
V2 Data Collection Pipeline Configuration.

All paths, thresholds, and caps for the per-creator worker pipeline.
Reads shared storage root from SHARED_ROOT env var (default: /mnt/shared).
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (clipwhy-v2/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Shared storage (GCS FUSE mount or local for testing) ────────────────────
SHARED_ROOT = Path(os.environ.get("SHARED_ROOT", "/mnt/shared"))

# Input
INPUT_DIR = SHARED_ROOT / "input"

# Progress tracking
PROGRESS_DIR = SHARED_ROOT / "progress"

# Raw downloads
RAW_DIR = SHARED_ROOT / "raw"
RAW_LONG_DIR = RAW_DIR / "long"
RAW_SHORTS_AUDIO_DIR = RAW_DIR / "shorts_audio"
RAW_SHORTS_VIDEO_DIR = RAW_DIR / "shorts_video"
RAW_CAPTIONS_DIR = RAW_DIR / "captions"

# 30s segments
SEGMENTS_DIR = SHARED_ROOT / "segments"

# Whisper transcripts
TRANSCRIPTS_DIR = SHARED_ROOT / "transcripts"
TRANSCRIPTS_SEGMENTS_DIR = TRANSCRIPTS_DIR / "segments"
TRANSCRIPTS_SHORTS_DIR = TRANSCRIPTS_DIR / "shorts"

# Pair matching results
PAIRS_DIR = SHARED_ROOT / "pairs"

# Engagement metadata
METADATA_DIR = SHARED_ROOT / "metadata"

# Labeled segments
LABELED_DIR = SHARED_ROOT / "labeled"

# Final merged output
FINAL_DIR = SHARED_ROOT / "final"

# Logs
LOGS_DIR = SHARED_ROOT / "logs"

# ── YouTube API keys ────────────────────────────────────────────────────────
YOUTUBE_API_KEYS = [
    v for k, v in sorted(os.environ.items())
    if k.startswith("YOUTUBE_API_KEY_") and v.strip()
]

# ── Discord ─────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ── Proxy (Cloudflare WARP on cloud VMs) ────────────────────────────────────
# 4 WARP containers per VM (warp0:1080, warp1:1081, warp2:1082, warp3:1083).
# Same exit IP but isolated restarts: if worker 2's proxy needs restart,
# workers 0, 1, 3 keep downloading uninterrupted.
PROXY_BASE_PORT = int(os.environ.get("PROXY_BASE_PORT", "1080"))
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")


def get_worker_proxy(worker_id: int) -> str:
    """Get proxy URL for a specific worker's WARP container.
    Returns empty string if PROXY_BASE_PORT is 0 or PROXY_URL is empty
    (disables proxy, for non-datacenter IPs like RunPod/home)."""
    proxy_url = os.environ.get("PROXY_URL", "")
    if proxy_url == "" or PROXY_BASE_PORT == 0:
        return ""
    port = PROXY_BASE_PORT + worker_id
    return f"socks5://{PROXY_HOST}:{port}"


def get_worker_container_name(worker_id: int) -> str:
    """Get Docker container name for a worker's WARP proxy."""
    return f"warp{worker_id}"

# ── Per-creator caps ────────────────────────────────────────────────────────
MAX_SHORTS_TO_LIST = 200        # shorts to list per creator (for baselines)
MAX_LONGS_TO_LIST = 50          # longs to list per creator
MAX_LONGS_FOR_CAPTIONS = 20     # longs to download captions for (matching)
MAX_PAIRS_PER_LONG_VIDEO = 5    # cap pairs per long video (diversity)

# ── Caption matching (pre-filter, NOT the label source) ─────────────────────
CAPTION_MATCH_THRESHOLD = 80    # fuzzy score, lower than V1's 90 since this
                                # is just a cheap filter before Whisper
CHUNK_SECONDS = 30              # VTT chunk size for timestamp localization

# ── Whisper ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_NAME = "base"     # "base" for accuracy (not "tiny")
WHISPER_DEVICE = "cuda"         # GPU; falls back to "cpu" if unavailable
WHISPER_MATCH_THRESHOLD = 80    # segment-level Whisper matching
HIGH_CONFIDENCE_THRESHOLD = 90
MIN_SHORT_WORDS = 10            # skip shorts with fewer Whisper words

# ── Segmentation ────────────────────────────────────────────────────────────
SEGMENT_DURATION_SEC = 30
MIN_OVERLAP_FOR_LABEL = 0.50    # 50% overlap for positive label

# ── Engagement & virality ───────────────────────────────────────────────────
MIN_SHORT_AGE_DAYS = 7
ENGAGEMENT_MULTIPLIER = 1.5     # starting multiplier (auto-adjusted)
VIEWS_PER_DAY_MULTIPLIER = 2.0  # starting multiplier (auto-adjusted)
VIRAL_SPLIT_MIN = 0.30
VIRAL_SPLIT_MAX = 0.70

# ── Downloads ───────────────────────────────────────────────────────────────
YTDLP_BIN = shutil.which("yt-dlp") or "yt-dlp"
DOWNLOAD_TIMEOUT_LONG_AUDIO = 2400    # 40 min (large files through WARP, 4 workers sharing bandwidth)
DOWNLOAD_TIMEOUT_LONG_VIDEO = 3600    # 60 min (720p video can be 500MB+)
DOWNLOAD_TIMEOUT_SHORT = 600          # 10 min
DOWNLOAD_TIMEOUT_CAPTION = 30         # 30 sec
MAX_DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_BACKOFF = [5, 15, 30]  # seconds between retries

# ── Concurrency ─────────────────────────────────────────────────────────────
WORKERS_PER_VM = 4
GPU_LOCK_PATH = Path("/tmp/clipwhy_gpu.lock")  # local per-VM, NOT shared

# ── Creator criteria (from V2 discovery, kept here for reference) ───────────
MIN_LONG_VIDEO_DURATION_SEC = 420   # 7 minutes
MAX_SHORT_DURATION_SEC = 60

# ── Notifications ───────────────────────────────────────────────────────────
STATUS_INTERVAL_SEC = 1800          # Discord summary every 30 min


def ensure_directories():
    """Create all output directories on shared storage."""
    for d in [
        INPUT_DIR, PROGRESS_DIR,
        RAW_LONG_DIR, RAW_SHORTS_AUDIO_DIR, RAW_SHORTS_VIDEO_DIR, RAW_CAPTIONS_DIR,
        SEGMENTS_DIR, TRANSCRIPTS_SEGMENTS_DIR, TRANSCRIPTS_SHORTS_DIR,
        PAIRS_DIR, METADATA_DIR, LABELED_DIR, FINAL_DIR, LOGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_video_item(item: dict) -> dict:
    """Flatten a raw YouTube API video item into a clean dict.

    Captures all useful fields from snippet, contentDetails, and statistics
    for later analysis. Fields not needed by the pipeline are still saved
    to metadata CSVs so they can be used for future research.
    """
    import re
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})
    thumbnails = snippet.get("thumbnails", {})

    # Parse ISO 8601 duration (PT10M30S -> 630)
    iso_dur = content.get("duration", "")
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_dur or "")
    if match:
        h, m, s = (int(g or 0) for g in match.groups())
        duration_seconds = h * 3600 + m * 60 + s
    else:
        duration_seconds = 0

    # Get best thumbnail URL
    thumb = (thumbnails.get("high") or thumbnails.get("medium")
             or thumbnails.get("default") or {})

    return {
        # Core (used by pipeline)
        "video_id": item.get("id", ""),
        "title": snippet.get("title", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration_seconds": duration_seconds,
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        # Snippet extras (useful for analysis)
        "description": snippet.get("description", ""),
        "tags": "|".join(snippet.get("tags", [])),  # pipe-separated
        "category_id": snippet.get("categoryId", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "thumbnail_url": thumb.get("url", ""),
        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
        # Content details extras
        "definition": content.get("definition", ""),  # hd or sd
        "has_captions": content.get("caption", "false"),
    }
