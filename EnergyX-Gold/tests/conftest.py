"""Shared fixtures for the independent checks.

`raw` reads BDG2's own files with pandas: no snapshot, no DuckDB, no EnergyX code. The CSVs
are cached as Parquet in .cache/raw/ after the first read, with the same values.
- raw.electricity_cleaned: wide, one column per building, indexed by local timestamp. NaN is missing.
- raw.electricity: the same shape, before BDG2's cleaning (zeros and outliers still in).
- raw.weather: long, one row per site and hour, with a local timestamp column.
- raw.metadata: BDG2_metadata.csv, one row per building, with Site name, City, Country and UID.

`gold(task_id)` runs a task through the runner once per session.
"""

from __future__ import annotations

import energyx_gold  # noqa: F401  (first, so its one-thread BLAS setting applies before numpy loads)

import functools
from pathlib import Path

import pandas as pd
import pytest

from energyx_gold.core.config import REPO_ROOT, load_config
from energyx_gold.core.context import GoldContext
from energyx_gold.core.loader import get_task
from energyx_gold.core.runner import run_task

CACHE = REPO_ROOT / ".cache" / "raw"


def _cached_csv(source: Path, name: str, **read_csv) -> pd.DataFrame:
    cached = CACHE / f"{name}.parquet"
    if cached.exists() and cached.stat().st_mtime >= source.stat().st_mtime:
        return pd.read_parquet(cached)
    frame = pd.read_csv(source, **read_csv)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached)
    return frame


class RawBDG2:
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir

    def _wide(self, name: str) -> pd.DataFrame:
        return _cached_csv(self.source_dir / f"{name}.csv", name,
                           parse_dates=["timestamp"], index_col="timestamp")

    @functools.cached_property
    def electricity_cleaned(self) -> pd.DataFrame:
        return self._wide("electricity_cleaned")

    @functools.cached_property
    def electricity(self) -> pd.DataFrame:
        return self._wide("electricity")

    @functools.cached_property
    def weather(self) -> pd.DataFrame:
        return _cached_csv(self.source_dir / "weather.csv", "weather", parse_dates=["timestamp"])

    @functools.cached_property
    def metadata(self) -> pd.DataFrame:
        return _cached_csv(self.source_dir / "BDG2_metadata.csv", "metadata")


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def raw(config) -> RawBDG2:
    return RawBDG2(config.bdg2_source_dir)


@pytest.fixture(scope="session")
def context(config) -> GoldContext:
    return GoldContext(config)


@pytest.fixture(scope="session")
def gold(context):
    @functools.cache
    def run(task_id: str) -> dict:
        return run_task(get_task(task_id), context)
    return run
