"""Package version resolution.

Priority:
  1. ``SPECTRAL_SR_VERSION`` env var (set at Docker build via ``APP_VERSION``)
  2. Installed package metadata from ``pyproject.toml``
  3. ``0.0.0+unknown`` when running from an unpacked source tree
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as pkg_version


def _resolve_version() -> str:
    env = os.environ.get("SPECTRAL_SR_VERSION", "").strip()
    if env:
        return env
    try:
        return pkg_version("spectral-sr")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()
