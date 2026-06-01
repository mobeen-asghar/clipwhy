"""Race-free creator claim mechanism using R2 conditional PUT.

The core primitive is atomic compare-and-swap (CAS) via S3 If-None-Match: "*".
Two pods that simultaneously try to claim the same creator will have exactly
one succeed; the loser gets a 412 and moves on.

Layout on R2:
  features_claims/{creator_id}.json    -- active claim, owned by one pod
  features_progress/{creator_id}_features_done.json  -- completion marker

Crash recovery:
  - Claim objects include heartbeat timestamp.
  - Pod renews heartbeat every CLAIM_RENEW_SECONDS while working.
  - If a claim's heartbeat is older than CLAIM_TTL_SECONDS, it is considered
    stale and is ignored; any pod may then re-claim via CAS.
  - Stale-claim overwriting uses plain PUT (idempotent last-write-wins) since
    the race window is short and the next heartbeat resolves ownership.

Why not a central orchestrator:
  - No single point of failure.
  - Pods can be added / removed at any time without reshuffling.
  - Exactly matches the "dynamic pool" the pods pull from.
"""
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

from . import config, r2_client

log = logging.getLogger("clipwhy.features.claims")


@dataclass
class Claim:
    creator_id: str
    vm_id: str
    claimed_at: float      # epoch seconds (UTC)
    heartbeat: float       # epoch seconds (UTC); renewed during work
    pid: int = 0           # process id, for diagnostics

    def is_stale(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return (now - self.heartbeat) > config.CLAIM_TTL_SECONDS

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")


# ── R2 key helpers ──────────────────────────────────────────────────────────
def _claim_key(creator_id: str) -> str:
    return f"{config.R2_CLAIMS_PREFIX}/{creator_id}.json"


def _done_key(creator_id: str) -> str:
    return f"{config.R2_PROGRESS_PREFIX}/{creator_id}_features_done.json"


# ── Listing helpers ─────────────────────────────────────────────────────────
def list_all_creators() -> list[str]:
    """Derive the canonical creator list from labeled CSVs on R2."""
    keys = r2_client.list_keys(config.R2_LABELED_PREFIX + "/")
    creators: set[str] = set()
    for k in keys:
        name = os.path.basename(k)
        if name.endswith("_segments.csv"):
            creators.add(name[: -len("_segments.csv")])
    return sorted(creators)


def list_done_creators() -> set[str]:
    keys = r2_client.list_keys(config.R2_PROGRESS_PREFIX + "/")
    done: set[str] = set()
    for k in keys:
        name = os.path.basename(k)
        suffix = "_features_done.json"
        if name.endswith(suffix):
            done.add(name[: -len(suffix)])
    return done


def list_active_claims() -> dict[str, Claim]:
    """Return {creator_id: Claim} for claims whose heartbeat is fresh."""
    keys = r2_client.list_keys(config.R2_CLAIMS_PREFIX + "/")
    now = time.time()
    active: dict[str, Claim] = {}
    for key in keys:
        name = os.path.basename(key)
        if not name.endswith(".json"):
            continue
        creator_id = name[:-len(".json")]
        blob = r2_client.get(key)
        if blob is None:
            continue
        try:
            data = json.loads(blob)
            claim = Claim(
                creator_id=data["creator_id"],
                vm_id=data["vm_id"],
                claimed_at=float(data["claimed_at"]),
                heartbeat=float(data["heartbeat"]),
                pid=int(data.get("pid", 0)),
            )
        except Exception as e:
            log.warning("Malformed claim %s: %s", key, e)
            continue
        if not claim.is_stale(now):
            active[creator_id] = claim
    return active


# ── Claim lifecycle ─────────────────────────────────────────────────────────
def try_claim(creator_id: str, vm_id: str) -> bool:
    """Attempt atomic CAS claim. Returns True on success."""
    now = time.time()
    claim = Claim(
        creator_id=creator_id,
        vm_id=vm_id,
        claimed_at=now,
        heartbeat=now,
        pid=os.getpid(),
    )
    # Fast path: conditional PUT.
    if r2_client.put_if_absent(_claim_key(creator_id), claim.to_bytes()):
        return True

    # Someone holds it. Check if stale; if yes, override with plain PUT.
    blob = r2_client.get(_claim_key(creator_id))
    if blob is None:
        # Claim disappeared between the CAS and the GET; retry CAS once.
        return r2_client.put_if_absent(_claim_key(creator_id), claim.to_bytes())
    try:
        existing = json.loads(blob)
        existing_claim = Claim(
            creator_id=existing["creator_id"],
            vm_id=existing["vm_id"],
            claimed_at=float(existing["claimed_at"]),
            heartbeat=float(existing["heartbeat"]),
            pid=int(existing.get("pid", 0)),
        )
    except Exception:
        # Corrupted claim; overwrite.
        r2_client.put(_claim_key(creator_id), claim.to_bytes())
        return True

    if existing_claim.is_stale():
        log.info(
            "[%s] Attempting to override stale claim on %s (was %s, heartbeat %ds old)",
            vm_id, creator_id, existing_claim.vm_id,
            int(time.time() - existing_claim.heartbeat),
        )
        # IMPORTANT: use delete + CAS instead of plain PUT.
        # If we plain-PUT, two pods both seeing the same stale claim would both
        # succeed and both think they own it. Delete + CAS forces the atomic
        # primitive to resolve which pod actually gets it.
        r2_client.delete(_claim_key(creator_id))
        return r2_client.put_if_absent(_claim_key(creator_id), claim.to_bytes())

    return False


def claim_next(vm_id: str) -> Optional[str]:
    """Pick an available creator and claim it atomically.

    Returns the creator_id on success, None when the pool is empty.
    Randomises candidate order so N pods don't serialise on the same first id.
    """
    all_creators = list_all_creators()
    done = list_done_creators()
    active = list_active_claims()

    candidates = [c for c in all_creators if c not in done and c not in active]
    if not candidates:
        return None

    random.shuffle(candidates)

    for creator_id in candidates:
        if try_claim(creator_id, vm_id):
            log.info("[%s] Claimed %s", vm_id, creator_id)
            return creator_id
        log.debug("[%s] Lost race for %s, trying next", vm_id, creator_id)

    return None


def renew(creator_id: str, vm_id: str) -> bool:
    """Refresh heartbeat on a claim we own — atomically.

    Uses GET-with-ETag + If-Match conditional PUT so that if another pod
    has taken over between our GET and PUT (e.g., we were stalled long
    enough to look stale), our PUT fails with 412 instead of silently
    clobbering their claim (which would cause double-processing).
    """
    blob, etag = r2_client.get_with_etag(_claim_key(creator_id))
    if blob is None or etag is None:
        return False
    try:
        data = json.loads(blob)
        if data.get("vm_id") != vm_id:
            return False
        data["heartbeat"] = time.time()
        ok = r2_client.put_if_match(
            _claim_key(creator_id),
            json.dumps(data).encode("utf-8"),
            etag,
        )
        if not ok:
            log.warning(
                "[%s] renew rejected by R2 (claim was modified) for %s; we lost ownership",
                vm_id, creator_id,
            )
        return ok
    except Exception as e:
        log.warning("[%s] renew failed for %s: %s", vm_id, creator_id, e)
        return False


def release(creator_id: str, vm_id: str) -> None:
    """Delete our claim, but only if we still own it."""
    blob = r2_client.get(_claim_key(creator_id))
    if blob is None:
        return
    try:
        data = json.loads(blob)
        if data.get("vm_id") == vm_id:
            r2_client.delete(_claim_key(creator_id))
            log.info("[%s] Released %s", vm_id, creator_id)
    except Exception:
        pass


def mark_done(creator_id: str, vm_id: str, stats: dict) -> None:
    """Write completion marker and release the claim."""
    body = {
        "creator_id": creator_id,
        "vm_id": vm_id,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **stats,
    }
    r2_client.put(_done_key(creator_id), json.dumps(body).encode("utf-8"))
    release(creator_id, vm_id)


# ── Pool summary (for CLI status) ───────────────────────────────────────────
def pool_status() -> dict:
    all_c = list_all_creators()
    done = list_done_creators()
    active = list_active_claims()
    done_s = set(done)
    active_s = set(active.keys())
    pending = [c for c in all_c if c not in done_s and c not in active_s]
    return {
        "total": len(all_c),
        "done": len(done),
        "active": len(active),
        "pending": len(pending),
        "active_claims": [
            {
                "creator_id": cid,
                "vm_id": c.vm_id,
                "age_seconds": int(time.time() - c.claimed_at),
                "heartbeat_age_seconds": int(time.time() - c.heartbeat),
            }
            for cid, c in sorted(active.items())
        ],
    }
