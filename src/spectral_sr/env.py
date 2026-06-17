"""Environment variable names for spectral-sr.

All application env vars are prefixed with ``SPECTRAL_RECONSTRUCTION_``.
"""

from __future__ import annotations

import os

_PREFIX = "SPECTRAL_RECONSTRUCTION_"


def _var(name: str) -> str:
    return f"{_PREFIX}{name}"


R2_ACCESS_KEY_ID = _var("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _var("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = _var("R2_ENDPOINT_URL")
R2_BUCKET_NAME = _var("R2_BUCKET_NAME")
R2_BUCKET_NAME_EXPERIMENTS = _var("R2_BUCKET_NAME_EXPERIMENTS")

HOST = _var("HOST")
PORT = _var("PORT")
WORKERS = _var("WORKERS")
LOG_LEVEL = _var("LOG_LEVEL")
API_KEY = _var("API_KEY")
SPECTRAL_SR_VERSION = _var("SPECTRAL_SR_VERSION")


def getenv(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
