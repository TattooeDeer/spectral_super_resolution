"""Cloudflare R2 / S3-compatible storage utilities.

Credentials are resolved in this priority order:
  1. Explicitly passed R2Config object (from API payload or CLI flags)
  2. Environment variables (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, …)
  3. .env file in the current working directory (loaded automatically at import)

No credentials are ever hard-coded here.

r2:// path convention
---------------------
Any path argument that begins with ``r2://`` is treated as an R2 key.
Use ``resolve_path()`` to transparently download it to a local temp file
before passing it to PyTorch / numpy loaders.

Example:
    local = resolve_path("r2://models/AEHG_150_100_75.pt", cfg)
    model.load_state_dict(torch.load(local))
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Load .env on import so env vars are available before anything else reads them
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # don't overwrite already-set env vars
except ImportError:
    pass  # dotenv is optional; real env vars still work fine


# ---------------------------------------------------------------------------
# R2Config
# ---------------------------------------------------------------------------

@dataclass
class R2Config:
    """
    Credentials and targeting configuration for Cloudflare R2.

    Fields default to the corresponding environment variables so callers
    only need to pass what differs from the environment.
    """
    access_key_id: str = field(default_factory=lambda: os.environ.get("R2_ACCESS_KEY_ID", ""))
    secret_access_key: str = field(default_factory=lambda: os.environ.get("R2_SECRET_ACCESS_KEY", ""))
    endpoint_url: str = field(default_factory=lambda: os.environ.get("R2_ENDPOINT_URL", ""))
    bucket: str = field(default_factory=lambda: os.environ.get("R2_BUCKET_NAME", "spectral-reconstruction-experiments"))

    def validate(self):
        """Raise EnvironmentError if any required field is missing."""
        missing = [f for f, v in [
            ("R2_ACCESS_KEY_ID / access_key_id", self.access_key_id),
            ("R2_SECRET_ACCESS_KEY / secret_access_key", self.secret_access_key),
            ("R2_ENDPOINT_URL / endpoint_url", self.endpoint_url),
        ] if not v]
        if missing:
            raise EnvironmentError(
                "Missing R2 credentials: " + ", ".join(missing) + "\n"
                "Set them in your .env file, as environment variables, "
                "or pass them in the request/CLI."
            )

    @classmethod
    def from_env(cls) -> "R2Config":
        """Convenience constructor: read everything from environment / .env."""
        return cls()

    def client(self):
        """Return a configured boto3 S3 client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required. Install with: uv pip install boto3")

        self.validate()
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def is_r2_path(path: str) -> bool:
    """Return True if ``path`` looks like ``r2://some/key``."""
    return isinstance(path, str) and path.startswith("r2://")


def r2_key_from_path(path: str) -> str:
    """Strip the ``r2://`` prefix to get the raw key."""
    return path[len("r2://"):]


def resolve_path(path: str, cfg: Optional[R2Config] = None,
                 tmp_dir: Optional[str] = None) -> str:
    """
    Resolve a path that may be local or an ``r2://`` reference.

    If the path is a local path it is returned unchanged.
    If it starts with ``r2://`` the object is downloaded to a temporary
    file and that local path is returned.

    Args:
        path:    Local filesystem path or ``r2://key``
        cfg:     R2Config to use for download (falls back to env vars)
        tmp_dir: Optional directory for temp files (defaults to system temp)

    Returns:
        Local filesystem path string
    """
    if not is_r2_path(path):
        return path

    key = r2_key_from_path(path)
    cfg = cfg or R2Config.from_env()

    suffix = Path(key).suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir)
    tmp.close()

    logger.info("Downloading r2://%s -> %s", key, tmp.name)
    cfg.client().download_file(cfg.bucket, key, tmp.name)
    logger.info("Download complete: %s", tmp.name)
    return tmp.name


def resolve_directory(r2_prefix: str, cfg: Optional[R2Config] = None,
                      local_dir: Optional[str] = None) -> str:
    """
    Download all objects under an R2 prefix to a local directory.

    Useful when dataset patches live in R2 and you want to point
    SpectralDataset_npy at a local folder.

    Args:
        r2_prefix: Key prefix inside the bucket (e.g. "data/hyperion_train_npy")
        cfg:       R2Config (falls back to env vars)
        local_dir: Where to put the files (defaults to a new temp dir)

    Returns:
        Local directory path string
    """
    cfg = cfg or R2Config.from_env()
    client = cfg.client()

    if local_dir is None:
        local_dir = tempfile.mkdtemp()
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Paginate over all objects with the prefix
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=cfg.bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Preserve sub-directory structure relative to the prefix
            relative = key[len(r2_prefix):].lstrip("/")
            local_file = local_dir_path / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading r2://%s -> %s", key, local_file)
            client.download_file(cfg.bucket, key, str(local_file))
            downloaded += 1

    logger.info("Downloaded %d files from r2://%s to %s", downloaded, r2_prefix, local_dir)
    return local_dir


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_file(local_path: Path, r2_key: str,
                cfg: Optional[R2Config] = None) -> str:
    """Upload a local file to R2 and return its key."""
    cfg = cfg or R2Config.from_env()
    local_path = Path(local_path)
    logger.info("Uploading %s -> r2://%s", local_path, r2_key)
    cfg.client().upload_file(str(local_path), cfg.bucket, r2_key)
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_bytes(data: bytes, r2_key: str,
                 content_type: str = "application/octet-stream",
                 cfg: Optional[R2Config] = None) -> str:
    """Upload raw bytes to R2."""
    cfg = cfg or R2Config.from_env()
    logger.info("Uploading bytes -> r2://%s", r2_key)
    cfg.client().put_object(
        Bucket=cfg.bucket,
        Key=r2_key,
        Body=io.BytesIO(data),
        ContentType=content_type,
    )
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_json(obj: dict, r2_key: str,
                cfg: Optional[R2Config] = None) -> str:
    """Upload a Python dict as a pretty-printed JSON file to R2."""
    data = json.dumps(obj, indent=2).encode("utf-8")
    return upload_bytes(data, r2_key, content_type="application/json", cfg=cfg)


def upload_directory(local_dir: Path, r2_prefix: str,
                     cfg: Optional[R2Config] = None,
                     extensions: tuple = (".pt", ".json", ".png", ".npy", ".csv")) -> List[str]:
    """
    Recursively upload all files under ``local_dir`` to R2.

    Args:
        local_dir:  Local source directory
        r2_prefix:  Key prefix (folder) inside the bucket
        cfg:        R2Config (falls back to env vars)
        extensions: Whitelist of file extensions (``None`` = all)

    Returns:
        List of uploaded R2 keys
    """
    cfg = cfg or R2Config.from_env()
    local_dir = Path(local_dir)
    uploaded: List[str] = []

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        relative = file_path.relative_to(local_dir)
        r2_key = f"{r2_prefix}/{relative}".replace("\\", "/")
        upload_file(file_path, r2_key, cfg=cfg)
        uploaded.append(r2_key)

    return uploaded


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_experiments(cfg: Optional[R2Config] = None) -> List[str]:
    """Return top-level experiment folder names in the bucket."""
    cfg = cfg or R2Config.from_env()
    response = cfg.client().list_objects_v2(Bucket=cfg.bucket, Delimiter="/")
    return [cp["Prefix"].rstrip("/") for cp in response.get("CommonPrefixes", [])]


def list_keys(prefix: str = "", cfg: Optional[R2Config] = None) -> List[str]:
    """List all object keys under an optional prefix."""
    cfg = cfg or R2Config.from_env()
    paginator = cfg.client().get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=cfg.bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys
