"""
YouTube Data API v3 wrapper with smart key rotation.

- Round-robin: spreads requests across all keys evenly
- Cooldown: exhausted keys are sidelined with a timestamp
- Recovery: checks hourly if exhausted keys are back (quota resets midnight PT)
- Never crash: if all keys are down, waits and retries periodically
- Status reports: periodic Discord/log updates for cloud monitoring
"""

import re
import time
import logging
from datetime import datetime, timezone

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger("clipwhy.api")

# YouTube API quota resets at midnight Pacific Time (UTC-7 PDT / UTC-8 PST)
RECOVERY_CHECK_SEC = 3600     # try exhausted keys every 1 hour
WAIT_POLL_SEC = 600           # when all keys down, check every 10 min
STATUS_INTERVAL_SEC = 1800    # report API status every 30 min


class YouTubeAPI:

    def __init__(self, api_keys, notify=None):
        if not api_keys:
            raise ValueError("No API keys. Check .env file.")
        self.api_keys = api_keys
        self.notify = notify

        # Pre-build a client per key (avoids rebuild on rotation)
        self.clients = []
        for key in api_keys:
            self.clients.append(
                build("youtube", "v3", developerKey=key, cache_discovery=False)
            )

        self.current_idx = 0
        self.usage = {i: 0 for i in range(len(api_keys))}
        self.exhausted = {}           # idx -> exhausted_at (datetime UTC)
        self._last_recovery_check = 0
        self._last_status_report = time.time()

        log.info("YouTube API: %d keys loaded", len(api_keys))

    # ── Key management ────────────────────────────────────────────────────────

    @property
    def client(self):
        return self.clients[self.current_idx]

    def _advance(self):
        """Round-robin to next active key. Returns True if one is available."""
        for _ in range(len(self.api_keys)):
            self.current_idx = (self.current_idx + 1) % len(self.api_keys)
            if self.current_idx not in self.exhausted:
                return True
        return False

    def _mark_exhausted(self, idx):
        """Mark a key as exhausted with timestamp."""
        self.exhausted[idx] = datetime.now(timezone.utc)
        active = len(self.api_keys) - len(self.exhausted)
        msg = (f"Key {idx + 1}/{len(self.api_keys)} quota exceeded "
               f"({self.usage[idx]} units used). {active} keys remaining.")
        log.warning(msg)
        if self.notify:
            self.notify.send(msg)

    def _check_recovery(self):
        """Try to reactivate exhausted keys after cooldown."""
        if not self.exhausted:
            return

        now = time.time()
        if now - self._last_recovery_check < RECOVERY_CHECK_SEC:
            return
        self._last_recovery_check = now

        to_test = list(self.exhausted.keys())
        for idx in to_test:
            elapsed = (datetime.now(timezone.utc) - self.exhausted[idx]).total_seconds()
            if elapsed < RECOVERY_CHECK_SEC:
                continue

            # Try a cheap API call (1 unit) to test if key is back
            try:
                self.clients[idx].channels().list(
                    part="id", id="UC_x5XG1OV2P6uZZ5FSM9Ttw",
                ).execute()

                # Key is back
                del self.exhausted[idx]
                self.usage[idx] += 1
                active = len(self.api_keys) - len(self.exhausted)
                msg = f"Key {idx + 1} recovered! {active} keys active."
                log.info(msg)
                if self.notify:
                    self.notify.send(msg)

            except HttpError as e:
                if "quotaExceeded" in str(e):
                    # Still exhausted, update timestamp for next check window
                    self.exhausted[idx] = datetime.now(timezone.utc)
                    log.debug("Key %d still exhausted", idx + 1)
                else:
                    log.warning("Key %d recovery check error: %s", idx + 1, e)

    def _wait_for_recovery(self):
        """All keys exhausted. Block until at least one recovers."""
        total_used = sum(self.usage.values())
        msg = (f"All {len(self.api_keys)} keys exhausted "
               f"({total_used} total units used). "
               f"Waiting for recovery (checking every 10 min)...")
        log.warning(msg)
        if self.notify:
            self.notify.send(msg)

        while True:
            time.sleep(WAIT_POLL_SEC)

            # Force recovery check regardless of cooldown timer
            self._last_recovery_check = 0
            self._check_recovery()

            active_keys = [
                i for i in range(len(self.api_keys)) if i not in self.exhausted
            ]
            if active_keys:
                self.current_idx = active_keys[0]
                msg = (f"Key {active_keys[0] + 1} recovered! "
                       f"Resuming with {len(active_keys)} active keys.")
                log.info(msg)
                if self.notify:
                    self.notify.send(msg)
                return

            # Still waiting
            msg = (f"Still waiting. {len(self.exhausted)}/{len(self.api_keys)} "
                   f"exhausted. Next check in 10 min.")
            log.info(msg)
            if self.notify:
                self.notify.send(msg)

    # ── Status reporting ──────────────────────────────────────────────────────

    def _maybe_report_status(self):
        """Send periodic API key status for cloud monitoring."""
        now = time.time()
        if now - self._last_status_report < STATUS_INTERVAL_SEC:
            return
        self._last_status_report = now
        self.report_status()

    def report_status(self):
        """Send current API key status report."""
        active = len(self.api_keys) - len(self.exhausted)
        total_used = sum(self.usage.values())
        parts = [
            f"[API Status] Active: {active}/{len(self.api_keys)} keys",
            f"Total units: {total_used}",
        ]
        if self.exhausted:
            exhausted_keys = sorted(k + 1 for k in self.exhausted)
            parts.append(f"Exhausted: keys {exhausted_keys}")
        # Per-key breakdown
        per_key = ", ".join(
            f"k{i + 1}:{self.usage[i]}" for i in range(len(self.api_keys))
            if self.usage[i] > 0
        )
        if per_key:
            parts.append(f"Usage: {per_key}")

        msg = " | ".join(parts)
        log.info(msg)
        if self.notify:
            self.notify.send(msg)

    # ── Request execution ─────────────────────────────────────────────────────

    def _call(self, build_fn, cost=1):
        """
        Execute an API call with round-robin, auto-rotation, and recovery.

        build_fn: callable that returns an API request using self.client
        cost: quota units for this call type
        """
        self._maybe_report_status()
        self._check_recovery()

        for _ in range(len(self.api_keys) + 1):
            try:
                request = build_fn()
                result = request.execute()
                self.usage[self.current_idx] += cost
                self._advance()  # round-robin to next key
                return result

            except HttpError as e:
                if e.resp.status == 403 and "quotaExceeded" in str(e):
                    self._mark_exhausted(self.current_idx)
                    if not self._advance():
                        self._wait_for_recovery()
                    continue  # retry with new key

                elif e.resp.status in (500, 503):
                    time.sleep(2)
                    continue

                else:
                    raise

        return None

    # ── Public API methods ────────────────────────────────────────────────────

    def search(self, q, search_type="video", topic_id=None, max_results=50):
        def build_req():
            params = dict(
                part="snippet", q=q, type=search_type,
                maxResults=min(max_results, 50),
                relevanceLanguage="en",
            )
            if topic_id:
                params["topicId"] = topic_id
            return self.client.search().list(**params)

        result = self._call(build_req, cost=100)
        return result.get("items", []) if result else []

    def get_channel_details(self, channel_ids):
        all_items = []
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i + 50]

            def build_req(ids=batch):
                return self.client.channels().list(
                    part="snippet,statistics,contentDetails,topicDetails",
                    id=",".join(ids),
                )

            result = self._call(build_req, cost=1)
            if result:
                all_items.extend(result.get("items", []))
        return all_items

    def get_video_details(self, video_ids):
        all_items = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]

            def build_req(ids=batch):
                return self.client.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(ids),
                )

            result = self._call(build_req, cost=1)
            if result:
                all_items.extend(result.get("items", []))
        return all_items

    def list_playlist_videos(self, playlist_id, max_videos=200):
        video_ids = []
        next_page = None

        while len(video_ids) < max_videos:
            def build_req(pt=next_page):
                return self.client.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=pt,
                )

            result = self._call(build_req, cost=1)
            if not result:
                break
            for item in result.get("items", []):
                video_ids.append(item["contentDetails"]["videoId"])
            next_page = result.get("nextPageToken")
            if not next_page:
                break

        return video_ids[:max_videos]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_quota_status(self):
        return {
            f"key_{i + 1}": {
                "used": self.usage[i],
                "exhausted": i in self.exhausted,
            }
            for i in range(len(self.api_keys))
        }

    @staticmethod
    def parse_duration(iso_duration):
        match = re.match(
            r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or ""
        )
        if not match:
            return 0
        h, m, s = (int(g or 0) for g in match.groups())
        return h * 3600 + m * 60 + s
