"""Helpers shared by the category 9 questions (Q41 to Q45)."""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.context import GoldContext

DECIMALS = 6
DDOF = 1  # sample standard deviation, as stddev_samp in DuckDB and std(ddof=1) in pandas

READINGS_QUERY = """
SELECT building_id, timestamp_local, value
FROM electricity
WHERE site_id = $site_id
  AND building_id IN ({ids})
  AND value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end
ORDER BY building_id, timestamp_local
"""


def rounded(value) -> float:
    return round(float(value), DECIMALS)


def site_of(building_id: str) -> str:
    return building_id.split("_")[0]


def valid_readings(context: GoldContext, building_ids: list[str], start, end) -> pd.DataFrame:
    """Valid readings (value not null) from `start` (inclusive) to `end` (exclusive), site-local.

    Long format: building_id, timestamp_local, value, in time order per building. All buildings
    must share one site.
    """
    sites = {site_of(b) for b in building_ids}
    if len(sites) != 1:
        raise ValueError(f"buildings from several sites: {sorted(sites)}")
    names = [f"b{i}" for i in range(len(building_ids))]
    params = {"site_id": sites.pop(), "start": pd.Timestamp(start).to_pydatetime(),
              "end": pd.Timestamp(end).to_pydatetime()}
    params.update(dict(zip(names, sorted(building_ids))))
    query = READINGS_QUERY.format(ids=", ".join("$" + n for n in names))
    frame = context.sql(query, params)
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    frame["value"] = frame["value"].astype(float)
    return frame


def sigma_count(values: pd.Series, sigma: float) -> dict:
    """Readings strictly above mean + sigma * sd (ddof 1), with the statistics behind the count."""
    mean, sd = float(values.mean()), float(values.std(ddof=DDOF))
    threshold = mean + sigma * sd
    above = values[values > threshold]
    return {"count": int(len(above)), "valid_hours": int(len(values)), "mean": mean, "sd": sd,
            "threshold": threshold, "max": float(values.max()), "above": above}


def hour_label(timestamp) -> str:
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:00")
