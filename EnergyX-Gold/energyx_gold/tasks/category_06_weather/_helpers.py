"""Helpers shared by the category 6 tasks: 2017 readings paired with site weather.

A reading is valid when its value isn't null. A pair is an hour with both a valid reading for the
building and a valid weather value at its site, in site-local standard time. Temperature is the
site's hourly air_temperature and dew point is dew_temperature, both in °C.

The office cohort is BDG2 primaryspaceusage "Office" with at least 90% of 2017's 8,760 hours
valid. No floor-area condition applies here.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from scipy import stats

from energyx_gold.core.context import GoldContext

YEAR = 2017
YEAR_START = datetime(YEAR, 1, 1)
YEAR_END = datetime(YEAR + 1, 1, 1)  # exclusive
HOURS_IN_YEAR = 8760
MIN_VALID_SHARE = 0.90
USE_LABEL_COLUMN = "building_type_raw"  # BDG2 primaryspaceusage
OFFICE = "Office"
TEMPERATURE = "air_temperature"
DEW_POINT = "dew_temperature"
WEATHER_VARIABLES = (TEMPERATURE, DEW_POINT)
DECIMALS = 6

COHORT_QUERY = """
SELECT b.building_id
FROM buildings b JOIN electricity e USING (building_id)
WHERE b.building_type_raw = $use_label
  AND e.timestamp_local >= $start AND e.timestamp_local < $end AND e.value IS NOT NULL
GROUP BY 1
HAVING COUNT(e.value) >= $min_valid_hours
ORDER BY 1
"""

PAIRED_QUERY = """
SELECT e.building_id, e.timestamp_local, e.value, w.{variable}
FROM electricity e
JOIN weather w ON w.site_id = e.site_id AND w.timestamp_local = e.timestamp_local
WHERE e.building_id IN ({placeholders})
  AND e.timestamp_local >= ? AND e.timestamp_local < ?
  AND e.value IS NOT NULL AND w.{variable} IS NOT NULL
ORDER BY e.building_id, e.timestamp_local
"""


def rounded(value: float) -> float:
    return round(float(value), DECIMALS)


def office_cohort(context: GoldContext) -> list[str]:
    """Office buildings with at least 90% of 2017's hours valid, sorted."""
    rows = context.sql(COHORT_QUERY, {
        "use_label": OFFICE,
        "start": YEAR_START,
        "end": YEAR_END,
        "min_valid_hours": MIN_VALID_SHARE * HOURS_IN_YEAR,
    })
    return sorted(rows["building_id"].astype(str))


def paired(context: GoldContext, building_ids: list[str], variable: str) -> pd.DataFrame:
    """Hours of 2017 with both a valid reading and a valid site weather value.

    Columns: building_id, timestamp_local, value, <variable>. One row per building and hour.
    """
    if variable not in WEATHER_VARIABLES:
        raise ValueError(f"unknown weather variable {variable!r}")
    query = PAIRED_QUERY.format(variable=variable, placeholders=", ".join("?" for _ in building_ids))
    frame = context.sql(query, [*building_ids, YEAR_START, YEAR_END])
    if frame.duplicated(["building_id", "timestamp_local"]).any():
        raise ValueError("a building and hour appear more than once; the weather has duplicate hours")
    return frame


def pearson(frame: pd.DataFrame, variable: str) -> float:
    return float(stats.pearsonr(frame["value"].to_numpy(float), frame[variable].to_numpy(float)).statistic)


def spearman(frame: pd.DataFrame, variable: str) -> float:
    """Spearman rank correlation; ties get average ranks (scipy's default)."""
    return float(stats.spearmanr(frame["value"].to_numpy(float), frame[variable].to_numpy(float)).statistic)


STATISTICS = {"pearson": pearson, "spearman": spearman}


def correlations(frame: pd.DataFrame, variable: str, method: str) -> pd.DataFrame:
    """Per building: the correlation of value with `variable` and the paired hours, sorted by building_id."""
    rows = {
        building_id: {"correlation": STATISTICS[method](group, variable), "paired_hours": len(group)}
        for building_id, group in frame.groupby("building_id", sort=True)
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def screen(context: GoldContext, variable: str, method: str, threshold: float) -> dict:
    """The office cohort screened on |correlation| > threshold: result and evidence body."""
    cohort = office_cohort(context)
    scores = correlations(paired(context, cohort, variable), variable, method)
    if scores["correlation"].isna().any() or len(scores) != len(cohort):
        raise ValueError("a cohort building has no correlation")
    members = sorted(scores.index[scores["correlation"].abs() > threshold].astype(str))
    ordered = scores.sort_index()
    ordered = ordered.reindex(ordered["correlation"].abs().sort_values(ascending=False, kind="mergesort").index)
    rejected = ordered[ordered["correlation"].abs() <= threshold]
    evidence = {
        "cohort_size": len(cohort),
        "correlations": [
            {"building_id": str(b), "correlation": rounded(r["correlation"]),
             "paired_hours": int(r["paired_hours"])}
            for b, r in ordered.iterrows()
        ],
        "highest_rejected_abs_correlation": rounded(rejected["correlation"].abs().max()) if len(rejected) else None,
        "negative_correlations": int((scores["correlation"] < 0).sum()),
    }
    return {"members": members, "evidence": evidence}


def period_note() -> str:
    return f"{YEAR_START:%Y-%m-%d %H:%M} to {YEAR_END:%Y-%m-%d %H:%M} (end exclusive), site-local"


def coverage_note() -> str:
    return f"valid hours >= {MIN_VALID_SHARE} x {HOURS_IN_YEAR}"


def set_result(members: list[str]) -> dict:
    members = sorted({str(member) for member in members})
    return {"answer": members, "count": len(members)}
