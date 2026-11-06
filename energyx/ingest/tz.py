"""Timezone handling for EnergyX snapshots.

BDG2 timestamps are naive and, as established empirically, expressed in each
site's **local standard time** -- a fixed UTC offset with no daylight-saving
adjustment. Two independent observations support this:

1. The electricity file carries a *single* timestamp column shared by 19 sites
   across 4 countries. One column of real instants cannot be wall-clock local
   time for all of them at once.
2. The column is a perfect continuous hourly grid: 17,544 unique values, no
   duplicates, every gap exactly one hour. DST wall-clock would necessarily
   contain a repeated hour each autumn and a missing hour each spring.

Corroboration: mean air temperature at the UK and Irish sites peaks at 15:00,
consistent with GMT rather than BST/IST.

Consequence: converting with a DST-aware zone would inject shifts the data does
not contain. Each site therefore gets a **constant** UTC offset, resolved once at
ingest and stored in the `sites` catalog table as ``utc_offset_hours``. Nothing
downstream re-derives it. See .

Offset rule
-----------
The offset in effect on 1 January -- i.e. the northern-hemisphere winter offset.

This is deliberately *not* ``dst() == 0``. Under modern IANA data
``Europe/Dublin`` declares Irish Standard Time (UTC+1) as its standard and treats
winter GMT as negative DST, so a naive "standard offset" probe returns +1 for
Dublin -- contradicted by the temperature evidence above.

All 19 BDG2 sites are northern hemisphere. A BYOD dataset in the southern
hemisphere, or one that genuinely carries DST, must declare ``utc_offset_hours``
explicitly in its manifest; ``resolve_offset`` honours that override.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

WINTER_PROBE = (1, 15)  # month, day


@lru_cache(maxsize=64)
def winter_offset(tz_name: str, year: int = 2017) -> timedelta:
    """UTC offset in effect on 15 January of ``year`` for ``tz_name``."""
    month, day = WINTER_PROBE
    moment = datetime(year, month, day, 12, 0, tzinfo=ZoneInfo(tz_name))
    return moment.utcoffset() or timedelta(0)


def resolve_offset(tz_name: str, override_hours: float | None = None) -> timedelta:
    """Constant offset for a site, honouring an explicit manifest override."""
    if override_hours is not None:
        return timedelta(hours=float(override_hours))
    return winter_offset(tz_name)


def offset_hours(tz_name: str, override_hours: float | None = None) -> float:
    return resolve_offset(tz_name, override_hours).total_seconds() / 3600.0


def local_to_utc(naive: pd.Series, offset: timedelta) -> pd.Series:
    """Naive local timestamps -> tz-aware UTC, using a constant offset."""
    return (pd.to_datetime(naive) - offset).dt.tz_localize("UTC")


def utc_to_local(utc: pd.Series, offset: timedelta) -> pd.Series:
    """tz-aware UTC -> naive local timestamps, using a constant offset."""
    return pd.to_datetime(utc).dt.tz_convert("UTC").dt.tz_localize(None) + offset
