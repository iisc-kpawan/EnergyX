"""Hourly electricity readings."""

from __future__ import annotations

from datetime import datetime

from .sql import select_readings as _select


def select_readings(building_ids: list[str], start: datetime | None = None,
                    end: datetime | None = None) -> tuple[str, list]:
    return _select("electricity", "building_id", building_ids, start, end,
                   "timestamp_local, building_id, value, quality_flag")
