"""
GPU access mutex for multi-worker VMs.

Uses fcntl file locking so only one worker thread uses the GPU at a time.
The lock file is on local /tmp (NOT shared storage) because:
  - fcntl is unreliable over NFS/GCS FUSE
  - Each VM has its own GPU, so the lock is VM-local

If a thread crashes, the OS releases the lock when the file descriptor
is garbage collected (safer than threading.Lock which stays held).
"""

import fcntl
import logging
import time
from pathlib import Path

from . import config

log = logging.getLogger("clipwhy.gpu")


class GPULock:
    """File-based mutex for GPU access. One holder per VM at a time."""

    def __init__(self, lock_path: Path | None = None):
        self.lock_path = lock_path or config.GPU_LOCK_PATH
        self._fd = None
        self._acquired_at = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    def acquire(self):
        """Block until the GPU lock is acquired."""
        log.debug("Waiting for GPU lock...")
        self._fd = open(self.lock_path, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._acquired_at = time.time()
        log.debug("GPU lock acquired")

    def release(self):
        """Release the GPU lock."""
        if self._fd:
            held = time.time() - self._acquired_at if self._acquired_at else 0
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None
            self._acquired_at = None
            log.debug("GPU lock released (held %.1fs)", held)
