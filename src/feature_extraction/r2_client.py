"""R2 client.

boto3 S3 client configured for Cloudflare R2, plus rclone subprocess helpers
for bulk media transfers.

R2 supports S3 conditional PUT (If-None-Match) since late 2024, which is what
we rely on for race-free creator claims (see claims.py).
"""
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from . import config

log = logging.getLogger("clipwhy.features.r2")

_client = None
_client_lock = threading.Lock()


def get_s3():
    """Thread-safe lazy boto3 client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = boto3.client(
                    "s3",
                    endpoint_url=config.R2_ENDPOINT,
                    aws_access_key_id=config.R2_ACCESS_KEY,
                    aws_secret_access_key=config.R2_SECRET_KEY,
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 5, "mode": "adaptive"},
                        s3={"addressing_style": "path"},
                    ),
                )
    return _client


def put_if_absent(key: str, body: bytes) -> bool:
    """Atomic conditional PUT. Returns True if we created it, False if it existed.

    This is the core primitive for race-free claim acquisition. If two pods
    call put_if_absent on the same key simultaneously, exactly one will succeed
    and the other will get a 412 PreconditionFailed.
    """
    try:
        get_s3().put_object(
            Bucket=config.R2_BUCKET,
            Key=key,
            Body=body,
            IfNoneMatch="*",
        )
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("PreconditionFailed", "ConditionalRequestConflict") or status == 412:
            return False
        raise


def put(key: str, body: bytes) -> None:
    get_s3().put_object(Bucket=config.R2_BUCKET, Key=key, Body=body)


def get_with_etag(key: str) -> tuple[Optional[bytes], Optional[str]]:
    """Like get(), but also returns the ETag for downstream conditional PUT."""
    try:
        obj = get_s3().get_object(Bucket=config.R2_BUCKET, Key=key)
        return obj["Body"].read(), obj.get("ETag")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "NoSuchBucket", "404"):
            return None, None
        raise


def put_if_match(key: str, body: bytes, etag: str) -> bool:
    """Conditional PUT: only succeeds if the current object has this ETag.

    Used by claims.renew to guarantee we don't clobber a newer claim that
    appeared between our GET and PUT (TOCTOU defense)."""
    try:
        get_s3().put_object(
            Bucket=config.R2_BUCKET,
            Key=key,
            Body=body,
            IfMatch=etag,
        )
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("PreconditionFailed", "ConditionalRequestConflict") or status == 412:
            return False
        raise


def get(key: str) -> Optional[bytes]:
    try:
        obj = get_s3().get_object(Bucket=config.R2_BUCKET, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "NoSuchBucket", "404"):
            return None
        raise


def head(key: str) -> Optional[dict]:
    try:
        return get_s3().head_object(Bucket=config.R2_BUCKET, Key=key)
    except ClientError:
        return None


def exists(key: str) -> bool:
    return head(key) is not None


def delete(key: str) -> None:
    try:
        get_s3().delete_object(Bucket=config.R2_BUCKET, Key=key)
    except ClientError as e:
        log.warning("delete failed for %s: %s", key, e)


def list_keys(prefix: str) -> list[str]:
    """List all keys under a prefix (paginated)."""
    paginator = get_s3().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=config.R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents") or []:
            keys.append(obj["Key"])
    return keys


# Default rclone subprocess timeout (seconds). One creator's bulk pull is
# ~3 GB; allow 30 min for the slowest R2/RunPod host before we give up.
RCLONE_TIMEOUT_SECONDS = 1800


def rclone_copy_down(remote_prefix: str, local_path: Path, transfers: int = 8,
                     timeout: int = RCLONE_TIMEOUT_SECONDS) -> None:
    """Bulk pull an R2 prefix to a local directory."""
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone", "copy",
        f"r2:{config.R2_BUCKET}/{remote_prefix}",
        str(local_path),
        "--disable-http2", "--update",
        f"--transfers={transfers}",
        "--s3-no-check-bucket",
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def rclone_copy_up(local_path: Path, remote_prefix: str, transfers: int = 8,
                   timeout: int = RCLONE_TIMEOUT_SECONDS) -> None:
    """Push a local directory up to an R2 prefix."""
    cmd = [
        "rclone", "copy",
        str(local_path),
        f"r2:{config.R2_BUCKET}/{remote_prefix}",
        "--disable-http2", "--update",
        f"--transfers={transfers}",
        "--s3-no-check-bucket",
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def rclone_copyto_up(local_file: Path, remote_key: str,
                     timeout: int = 300) -> None:
    """Push a single local file to a specific R2 key."""
    cmd = [
        "rclone", "copyto",
        str(local_file),
        f"r2:{config.R2_BUCKET}/{remote_key}",
        "--disable-http2", "--update",
        "--s3-no-check-bucket",
    ]
    subprocess.run(cmd, check=True, timeout=timeout)
