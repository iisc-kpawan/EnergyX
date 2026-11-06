"""Hourly weather by site."""

from __future__ import annotations

from datetime import datetime

from .sql import select_readings


def select_weather(site_ids: list[str], start: datetime | None = None, end: datetime | None = None,
                   variables: list[str] | None = None) -> tuple[str, list]:
    for variable in variables or []:
        if not variable.isidentifier():
            raise ValueError(f"bad variable name {variable!r}")
    columns = "timestamp_local, site_id, " + (", ".join(variables) if variables else "* EXCLUDE (timestamp_local, site_id)")
    return select_readings("weather", "site_id", site_ids, start, end, columns)
