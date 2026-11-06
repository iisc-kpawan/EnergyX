"""Run-scoped artifact store backing the handle pattern.

Frames live on disk under ``runs/<run_id>/artifacts/`` and are addressed by
handle. The store is bounded in two ways so that a 6,000-question benchmark
sweep cannot fill a disk that is already near capacity: a per-run byte cap, and
a TTL sweep across old runs.
"""
from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import PATHS

DEFAULT_MAX_BYTES = 2 * 1024**3     # 2 GiB per run
DEFAULT_TTL_SECONDS = 24 * 3600


@dataclass
class ArtifactStore:
    run_id: str
    root: Path
    max_bytes: int = DEFAULT_MAX_BYTES

    @classmethod
    def for_run(cls, run_id: str | None = None, root: Path | None = None,
                max_bytes: int = DEFAULT_MAX_BYTES) -> "ArtifactStore":
        run_id = run_id or f"r_{uuid.uuid4().hex[:10]}"
        base = Path(root or PATHS.runs) / run_id / "artifacts"
        base.mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, root=base, max_bytes=max_bytes)

    # -- writing -----------------------------------------------------------
    def put(self, df: pd.DataFrame, name: str = "result") -> str:
        handle = f"{name[:24]}_{uuid.uuid4().hex[:8]}"
        path = self.root / f"{handle}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        self._enforce_cap()
        return handle

    # -- reading -----------------------------------------------------------
    def get(self, handle: str) -> pd.DataFrame:
        path = self.root / f"{handle}.parquet"
        if not path.exists():
            raise KeyError(f"unknown handle: {handle}")
        return pd.read_parquet(path)

    def get_or_none(self, handle: str | None) -> pd.DataFrame | None:
        """Lookup that declines rather than raises, for optional inputs."""
        if not handle or not self.exists(handle):
            return None
        return self.get(handle)

    def exists(self, handle: str) -> bool:
        return (self.root / f"{handle}.parquet").exists()

    def handles(self) -> list[str]:
        """Every table this run has stored, newest last."""
        return [p.stem for p in
                sorted(self.root.glob("*.parquet"), key=lambda q: q.stat().st_mtime)]

    def describe(self, handle: str) -> dict:
        path = self.root / f"{handle}.parquet"
        return {"handle": handle, "bytes": path.stat().st_size if path.exists() else 0}

    @property
    def bytes_used(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.parquet"))

    # -- housekeeping ------------------------------------------------------
    def _enforce_cap(self) -> None:
        """Evict oldest artifacts once the run exceeds its byte budget."""
        files = sorted(self.root.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > self.max_bytes and len(files) > 1:
            victim = files.pop(0)
            total -= victim.stat().st_size
            victim.unlink(missing_ok=True)

    @staticmethod
    def sweep(root: Path | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
        """Delete run directories older than the TTL. Returns how many went."""
        base = Path(root or PATHS.runs)
        if not base.exists():
            return 0
        cutoff = time.time() - ttl_seconds
        removed = 0
        for run_dir in base.iterdir():
            if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                shutil.rmtree(run_dir, ignore_errors=True)
                removed += 1
        return removed
