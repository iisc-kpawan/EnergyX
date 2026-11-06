"""Where a gold result came from: the data, the code and the library versions."""

from __future__ import annotations

import functools
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .config import REPO_ROOT, Config

PACKAGES = ("duckdb", "pandas", "pyarrow", "numpy", "pydantic", "pyoxigraph", "scikit-learn", "sktime")
DATA_DIRS = ("catalog", "electricity", "weather")


@functools.cache
def snapshot_sha256(snapshot_dir: Path) -> str:
    """SHA-256 over the snapshot's Parquet files: each file's path and content hash, in path order.

    MANIFEST.json is left out because it records when the snapshot was built.
    """
    digest = hashlib.sha256()
    files = sorted(p for d in DATA_DIRS for p in (snapshot_dir / d).rglob("*.parquet"))
    for path in files:
        digest.update(str(path.relative_to(snapshot_dir)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@functools.cache
def _git_state() -> tuple[str | None, bool | None]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


@functools.cache
def _packages() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def collect(config: Config) -> dict:
    lock = REPO_ROOT / "uv.lock"
    commit, dirty = _git_state()
    return {
        "data_version": config.data_version,
        "snapshot_sha256": snapshot_sha256(config.snapshot_dir),
        "code_commit": commit,
        "code_dirty": dirty,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": sys.version.split()[0],
        "packages": _packages(),
        "platform": platform.platform(),
    }
