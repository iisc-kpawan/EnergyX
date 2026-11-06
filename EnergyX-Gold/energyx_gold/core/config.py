"""Load config/paths.yaml, or the file ENERGYX_GOLD_CONFIG names."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paths.yaml"


class Period(BaseModel):
    """The snapshot period in site-local standard time. `end` is the last hour, inclusive."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @property
    def expected_hours(self) -> int:
        return int((self.end - self.start).total_seconds() // 3600) + 1


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_version: str
    snapshot_dir: Path
    bdg2_source_dir: Path
    questions_xlsx: Path
    period: Period
    seed: int = 0

    @classmethod
    def _resolve(cls, value: str | None) -> Path | None:
        """`~` expands to the home folder; a relative path is relative to the repo root."""
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def graph_dir(self) -> Path:
        return self.snapshot_dir / "graph"


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path or os.environ.get("ENERGYX_GOLD_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    return Config.model_validate(yaml.safe_load(path.read_text()))
