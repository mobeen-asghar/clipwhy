"""
Download wrappers for yt-dlp with proxy support and retries.

Adapted from V1:
  - clipwhy-scraper/download_segment.py (download_audio, download_video)
  - clipwhy-scraper/rematch_whisper.py  (download_short_audio)
  - clipwhy-v2/src/data_collection/verify.py (_proxy_args pattern)
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger("clipwhy.download")

# Per-worker failure tracking for WARP restart detection
_worker_failures: dict[int, int] = {}  # worker_id -> consecutive failure count
_failure_lock = threading.Lock()
_WARP_RESTART_THRESHOLD = 3


def _restart_warp(worker_id: int):
    """Restart this worker's WARP Docker container to get a new IP."""
    container = config.get_worker_container_name(worker_id)
    proxy_url = config.get_worker_proxy(worker_id)
    log.warning("W%d: Restarting WARP container '%s'...", worker_id, container)
    try:
        subprocess.run(
            ["sudo", "docker", "restart", container],
            capture_output=True, text=True, timeout=30,
        )
        time.sleep(10)  # wait for WARP to reconnect
        # Verify new IP
        socks_url = proxy_url.replace("socks5://", "socks5h://")
        result = subprocess.run(
            ["curl", "-s", "--proxy", socks_url, "https://httpbin.org/ip"],
            capture_output=True, text=True, timeout=15,
        )
        log.info("W%d: WARP restarted. New IP: %s",
                 worker_id, result.stdout.strip()[:50])
    except Exception as e:
        log.warning("W%d: WARP restart failed: %s", worker_id, e)


def _run_ytdlp(args: list[str], timeout: int, worker_id: int = 0) -> bool:
    """Run yt-dlp with retries, backoff, and automatic WARP restart.

    Each worker uses its own WARP proxy (warp0:1080, warp1:1081, etc.)
    so restarts don't interfere with other workers.
    """
    proxy_url = config.get_worker_proxy(worker_id)
    proxy_args = ["--proxy", proxy_url] if proxy_url else []
    full_cmd = [
        config.YTDLP_BIN, "--no-warnings", "-q",
        "--remote-components", "ejs:github",
    ] + proxy_args + args

    for attempt in range(config.MAX_DOWNLOAD_RETRIES):
        try:
            subprocess.run(
                full_cmd, capture_output=True, text=True,
                timeout=timeout, check=True,
            )
            # Success: reset this worker's failure counter
            with _failure_lock:
                _worker_failures[worker_id] = 0
            return True
        except subprocess.TimeoutExpired:
            log.warning("  W%d: yt-dlp timeout (attempt %d/%d)",
                        worker_id, attempt + 1, config.MAX_DOWNLOAD_RETRIES)
        except subprocess.CalledProcessError as e:
            log.warning("  W%d: yt-dlp error (attempt %d/%d): %s",
                        worker_id, attempt + 1, config.MAX_DOWNLOAD_RETRIES,
                        e.stderr[:200] if e.stderr else str(e))

        if attempt < config.MAX_DOWNLOAD_RETRIES - 1:
            wait = config.DOWNLOAD_RETRY_BACKOFF[attempt]
            time.sleep(wait)

    # All retries failed: track and maybe restart this worker's WARP
    with _failure_lock:
        _worker_failures[worker_id] = _worker_failures.get(worker_id, 0) + 1
        if _worker_failures[worker_id] >= _WARP_RESTART_THRESHOLD:
            _restart_warp(worker_id)
            _worker_failures[worker_id] = 0

    return False


# ── Long video downloads ────────────────────────────────────────────────────


def download_long_audio(video_id: str, worker_id: int = 0) -> Path | None:
    """Download long video audio as WAV 16kHz mono. Returns path or None."""
    output = config.RAW_LONG_DIR / f"{video_id}.wav"
    if output.exists() and output.stat().st_size > 1000:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ytdlp(
        ["-x", "--audio-format", "wav",
         "--postprocessor-args", "-ar 16000 -ac 1",
         "-o", str(output),
         f"https://youtube.com/watch?v={video_id}"],
        timeout=config.DOWNLOAD_TIMEOUT_LONG_AUDIO,
        worker_id=worker_id,
    )
    return output if ok and output.exists() else None


def download_long_video(video_id: str, worker_id: int = 0) -> Path | None:
    """Download long video as MP4 720p. Returns path or None."""
    output = config.RAW_LONG_DIR / f"{video_id}.mp4"
    if output.exists() and output.stat().st_size > 1000:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ytdlp(
        ["-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
              "/best[height<=720][ext=mp4]/best",
         "--merge-output-format", "mp4",
         "-o", str(output),
         f"https://youtube.com/watch?v={video_id}"],
        timeout=config.DOWNLOAD_TIMEOUT_LONG_VIDEO,
        worker_id=worker_id,
    )
    return output if ok and output.exists() else None


# ── Short downloads ─────────────────────────────────────────────────────────


def download_short_audio(short_id: str, worker_id: int = 0) -> Path | None:
    """Download Short audio as WAV 16kHz mono. Returns path or None."""
    output = config.RAW_SHORTS_AUDIO_DIR / f"{short_id}.wav"
    if output.exists() and output.stat().st_size > 1000:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ytdlp(
        ["-x", "--audio-format", "wav",
         "--postprocessor-args", "-ar 16000 -ac 1",
         "-o", str(output),
         f"https://youtube.com/watch?v={short_id}"],
        timeout=config.DOWNLOAD_TIMEOUT_SHORT,
        worker_id=worker_id,
    )
    return output if ok and output.exists() else None


def download_short_video(short_id: str, worker_id: int = 0) -> Path | None:
    """Download Short video as MP4. Returns path or None."""
    output = config.RAW_SHORTS_VIDEO_DIR / f"{short_id}.mp4"
    if output.exists() and output.stat().st_size > 1000:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    ok = _run_ytdlp(
        ["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "--merge-output-format", "mp4",
         "-o", str(output),
         f"https://youtube.com/watch?v={short_id}"],
        timeout=config.DOWNLOAD_TIMEOUT_SHORT,
        worker_id=worker_id,
    )
    return output if ok and output.exists() else None


# ── Caption downloads ───────────────────────────────────────────────────────


def download_caption(video_id: str, worker_id: int = 0) -> Path | None:
    """Download YouTube auto-caption as VTT via WARP proxy with retries.
    Returns VTT path or None."""
    vtt = config.RAW_CAPTIONS_DIR / f"{video_id}.en.vtt"
    if vtt.exists() and vtt.stat().st_size > 100:
        return vtt

    vtt.parent.mkdir(parents=True, exist_ok=True)

    # Use proxy + retries (YouTube throttles concurrent caption requests
    # from GCP IPs even though captions are lightweight)
    proxy_url = config.get_worker_proxy(worker_id)
    proxy_args = ["--proxy", proxy_url] if proxy_url else []

    for attempt in range(config.MAX_DOWNLOAD_RETRIES):
        try:
            subprocess.run(
                [config.YTDLP_BIN, "--no-warnings", "-q",
                 "--remote-components", "ejs:github"]
                + proxy_args
                + ["--write-auto-sub", "--sub-lang", "en",
                   "--skip-download", "--sub-format", "vtt",
                   "-o", str(config.RAW_CAPTIONS_DIR / "%(id)s"),
                   f"https://youtube.com/watch?v={video_id}"],
                capture_output=True, text=True,
                timeout=60,
            )
            if vtt.exists() and vtt.stat().st_size > 100:
                return vtt
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        if attempt < config.MAX_DOWNLOAD_RETRIES - 1:
            time.sleep(config.DOWNLOAD_RETRY_BACKOFF[attempt])

    return None


# ── Duration probe ──────────────────────────────────────────────────────────


def get_duration(file_path: Path) -> float:
    """Get media duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(file_path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
