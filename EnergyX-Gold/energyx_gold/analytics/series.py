"""Load hourly readings for a cohort of buildings as a wide frame, and fill short gaps.

Cohort words in the questions map to BDG2 metadata:
- a building type ("education", "office", "parking", "entertainment/public assembly", ...) is
  `building_type_raw`, BDG2's `primaryspaceusage`;
- "US" and "UK" are `country_canonical` of the building's site. The anonymous sites (Bobcat, Eagle,
  Gator, Hog, Swan) have no country, so they are in neither.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..core.context import GoldContext


def cohort(context: GoldContext, *, with_readings: bool = True, **filters) -> list[str]:
    """Building IDs matching the metadata filters, sorted. Only buildings with a valid reading, by default."""
    frame = context.get_metadata(**filters)
    ids = sorted(frame["building_id"].astype(str))
    if not with_readings or not ids:
        return ids
    sites = sorted({b.split("_")[0] for b in ids})
    have = context.sql(
        f"SELECT DISTINCT building_id FROM electricity WHERE site_id IN ({', '.join('?' for _ in sites)}) "
        f"AND building_id IN ({', '.join('?' for _ in ids)}) AND value IS NOT NULL",
        [*sites, *ids],
    )
    return sorted(set(ids) & set(have["building_id"].astype(str)))


def hourly(context: GoldContext, building_ids: list[str], start: datetime | str, end: datetime | str) -> pd.DataFrame:
    """Valid hourly readings from `start` (inclusive) to `end` (exclusive), site-local time.

    One column per building, one row per hour of the period. NaN where there is no valid reading.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    index = pd.date_range(start, end, freq="h", inclusive="left", name="timestamp_local")
    if not building_ids:
        return pd.DataFrame(index=index)
    sites = sorted({b.split("_")[0] for b in building_ids})
    rows = context.sql(
        f"SELECT timestamp_local, building_id, value FROM electricity "
        f"WHERE site_id IN ({', '.join('?' for _ in sites)}) "
        f"AND building_id IN ({', '.join('?' for _ in building_ids)}) "
        f"AND value IS NOT NULL AND timestamp_local >= ? AND timestamp_local < ?",
        [*sites, *building_ids, start.to_pydatetime(), end.to_pydatetime()],
    )
    wide = rows.pivot(index="timestamp_local", columns="building_id", values="value")
    return wide.reindex(index=index, columns=sorted(building_ids)).astype(float)


def weather(context: GoldContext, site_id: str, start: datetime | str, end: datetime | str,
            variables: list[str]) -> pd.DataFrame:
    """Hourly weather for one site on the full hourly index of the period, site-local time."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    index = pd.date_range(start, end, freq="h", inclusive="left", name="timestamp_local")
    frame = context.get_weather([site_id], start.to_pydatetime(), end.to_pydatetime(), variables)
    frame = frame.drop_duplicates("timestamp_local").set_index("timestamp_local")[variables]
    return frame.reindex(index).astype(float)


def fill_gaps(frame: pd.DataFrame | pd.Series, limit: int = 24) -> pd.DataFrame | pd.Series:
    """Linear interpolation inside gaps of up to `limit` hours. Longer gaps and the edges stay NaN.

    Used only for model inputs. Errors are always measured against real readings.
    """
    def one(series: pd.Series) -> pd.Series:
        missing = series.isna()
        run_id = (missing != missing.shift()).cumsum()
        run_length = missing.groupby(run_id).transform("sum")
        filled = series.interpolate(method="linear", limit_area="inside")
        return filled.where(~(missing & (run_length > limit)))

    return frame.apply(one) if isinstance(frame, pd.DataFrame) else one(frame)
