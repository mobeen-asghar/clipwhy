"""Logging and Discord notifications."""

import time
import logging
from datetime import datetime

log = logging.getLogger("clipwhy.notify")


class Notifier:

    def __init__(self, webhook_url="", log_path=None):
        self.webhook_url = webhook_url
        self.log_path = log_path
        self._start = time.time()

    def _elapsed(self):
        m = int((time.time() - self._start) / 60)
        return f"{m // 60}h{m % 60}m"

    def _write_log(self, msg):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{self._elapsed()}] {msg}\n")

    def send(self, msg):
        log.info(msg)
        self._write_log(msg)
        if self.webhook_url:
            try:
                import requests
                requests.post(self.webhook_url, json={"content": msg}, timeout=5)
            except Exception:
                pass

    def progress(self, label, current, total, extra=""):
        pct = current / total * 100 if total else 0
        msg = f"[{label}] {current}/{total} ({pct:.1f}%)"
        if extra:
            msg += f" | {extra}"
        self.send(msg)

    def error(self, msg):
        self.send(f"ERROR: {msg}")

    def done(self, msg):
        self.send(f"DONE: {msg} (total time: {self._elapsed()})")
