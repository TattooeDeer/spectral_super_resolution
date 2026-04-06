"""Cloudflare R2 storage utilities for uploading experiment results.

R2 is S3-compatible, so we use boto3 with a custom endpoint URL.

Required environment variables:
    R2_ACCESS_KEY_ID     - Cloudflare R2 Access Key ID
    R2_SECRET_ACCESS_KEY - Cloudflare R2 Secret Access Key
    R2_ENDPOINT_URL      - https://<account_id>.r2.cloudflarestorage.com
    R2_BUCKET_NAME       - Bucket name (default: spectral-reconstruction-experiments)
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

R2_ENDPOINT_URL = os.getenv(
    "R2_ENDPOINT_URL",
    "https://db79fd90f3dfd68702afa1f74d455523.r2.cloudflarestorage.com",
)
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "spectral-reconstruction-experiments")


def _get_client():
    """Create a boto3 S3 client pointed at Cloudflare R2."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for R2 storage. Install with: uv pip install boto3")

    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise EnvironmentError(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set as environment variables."
        )

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def upload_file(local_path: Path, r2_key: str, bucket: str = R2_BUCKET_NAME) -> str:
    """
    Upload a local file to R2 and return its R2 key.

    Args:
        local_path: Path to the local file
        r2_key:     Destination key inside the bucket
        bucket:     Bucket name

    Returns:
        The full R2 key of the uploaded object
    """
    client = _get_client()
    local_path = Path(local_path)

    logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, r2_key)
    client.upload_file(str(local_path), bucket, r2_key)
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_bytes(data: bytes, r2_key: str, content_type: str = "application/octet-stream",
                 bucket: str = R2_BUCKET_NAME) -> str:
    """
    Upload raw bytes to R2.

    Args:
        data:         Bytes to upload
        r2_key:       Destination key inside the bucket
        content_type: MIME type
        bucket:       Bucket name

    Returns:
        The full R2 key of the uploaded object
    """
    client = _get_client()
    logger.info("Uploading bytes -> s3://%s/%s", bucket, r2_key)
    client.put_object(
        Bucket=bucket,
        Key=r2_key,
        Body=io.BytesIO(data),
        ContentType=content_type,
    )
    logger.info("Upload complete: %s", r2_key)
    return r2_key


def upload_json(obj: dict, r2_key: str, bucket: str = R2_BUCKET_NAME) -> str:
    """Upload a Python dict as a JSON file to R2."""
    data = json.dumps(obj, indent=2).encode("utf-8")
    return upload_bytes(data, r2_key, content_type="application/json", bucket=bucket)


def upload_directory(local_dir: Path, r2_prefix: str, bucket: str = R2_BUCKET_NAME,
                     extensions: tuple = (".pt", ".json", ".png", ".npy", ".csv")) -> list:
    """
    Recursively upload all files in a local directory to R2.

    Args:
        local_dir:  Local directory to upload
        r2_prefix:  Prefix (folder path) inside the bucket
        bucket:     Bucket name
        extensions: File extensions to upload (None = all files)

    Returns:
        List of uploaded R2 keys
    """
    local_dir = Path(local_dir)
    uploaded = []

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        relative = file_path.relative_to(local_dir)
        r2_key = f"{r2_prefix}/{relative}".replace("\\", "/")
        upload_file(file_path, r2_key, bucket=bucket)
        uploaded.append(r2_key)

    return uploaded


def list_experiments(bucket: str = R2_BUCKET_NAME) -> list:
    """
    List top-level experiment folders in the R2 bucket.

    Returns:
        List of experiment prefixes (folder names)
    """
    client = _get_client()
    response = client.list_objects_v2(Bucket=bucket, Delimiter="/")

    prefixes = [cp["Prefix"].rstrip("/") for cp in response.get("CommonPrefixes", [])]
    return prefixes


def public_url(r2_key: str, bucket: str = R2_BUCKET_NAME) -> str:
    """Return the public HTTPS URL for an R2 object (requires public bucket/rule)."""
    return f"{R2_ENDPOINT_URL}/{bucket}/{r2_key}"
