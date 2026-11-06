"""Load config/max.yaml, or the file ENERGYX_MAX_CONFIG names."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "max.yaml"


def resolve(value: str | Path) -> Path:
    """`~` expands to the home folder; a relative path is relative to the repo root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@functools.cache
def load() -> dict:
    path = Path(os.environ.get("ENERGYX_MAX_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    return yaml.safe_load(path.read_text())


def path(*keys: str) -> Path:
    """A path from the `paths` section, e.g. path("seeds", "team")."""
    value = load()["paths"]
    for key in keys:
        value = value[key]
    return resolve(value)


def setting(section: str, key: str):
    return load()[section][key]
