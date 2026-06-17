"""Package version resolution.

Priority:
  1. ``SPECTRAL_RECONSTRUCTION_SPECTRAL_SR_VERSION`` env var (set at Docker build via ``APP_VERSION``)
  2. Installed package metadata from ``pyproject.toml``
  3. ``0.0.0+unknown`` when running from an unpacked source tree
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version

from . import env


def _resolve_version() -> str:
    env_version = env.getenv(env.SPECTRAL_SR_VERSION).strip()
    if env_version:
        return env_version
    try:
        return pkg_version("spectral-sr")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()
