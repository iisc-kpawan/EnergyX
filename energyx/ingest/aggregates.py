"""Derived aggregates for query speed.

Raw hourly data stays authoritative; these tables exist so that a question like
"which buildings consumed most last month" need not scan 26M rows.

Two properties matter:

* **Local calendar boundaries.** A "day" is a local day at the building's site,
  not a UTC day. Aggregating in UTC would slice Tempe's days at 17:00 local.
  Local time comes from the ``electricity`` view, which derives it once from the
  site's constant offset -- this module never re-implements that shift.
* **Coverage awareness.** Every row reports the observations it was built from,
  so a partial month is never silently compared against a complete one.

``ZERO_MASKED`` readings are null in ``value`` and therefore excluded
from sums, but are counted separately so a tool that wants to treat masked zeros
as real zeros has the information to do so.
"""
from __future__ import annotations

from pathlib import Path

from ..access.db import connect
from ..config import PATHS

_METRICS = """
        COUNT(value)                                        AS n_observations,
        COUNT(*) FILTER (WHERE quality_flag = 'ZERO_MASKED') AS n_zero_masked,
        COUNT(*)                                            AS n_rows_present,
        SUM(value)                                          AS total_kwh,
        AVG(value)                                          AS mean_demand,
        MAX(value)                                          AS peak_demand,
        MIN(value)                                          AS min_demand,
        QUANTILE_CONT(value, 0.5)                           AS median_demand,
        ARG_MAX(EXTRACT(hour FROM timestamp_local), value)  AS peak_hour_local
"""

GRAINS = {
    "daily":   ("CAST(timestamp_local AS DATE)",          "24"),
    "monthly": ("DATE_TRUNC('month', timestamp_local)",   "DATEDIFF('hour', period_start, period_start + INTERVAL 1 MONTH)"),
    "yearly":  ("DATE_TRUNC('year', timestamp_local)",    "DATEDIFF('hour', period_start, period_start + INTERVAL 1 YEAR)"),
}


def build_aggregates(snapshot: Path | None = None) -> dict[str, int]:
    snapshot = Path(snapshot or PATHS.snapshot)
    out = snapshot / "aggregates"
    (out / "profiles").mkdir(parents=True, exist_ok=True)

    con = connect(snapshot)
    written: dict[str, int] = {}

    for grain, (bucket, expected) in GRAINS.items():
        target = out / grain
        target.mkdir(parents=True, exist_ok=True)
        path = target / "part-000.parquet"
        con.execute(f"""
            COPY (
                WITH base AS (
                    SELECT building_id, stream_id, site_id,
                           {bucket} AS period_start,
                           {_METRICS}
                    FROM electricity
                    GROUP BY 1, 2, 3, 4
                )
                SELECT *, '{grain}' AS grain,
                       {expected} AS expected_observations,
                       n_observations::DOUBLE / NULLIF({expected}, 0) AS coverage_fraction
                FROM base ORDER BY building_id, period_start
            ) TO '{path}' (FORMAT parquet, COMPRESSION zstd)
        """)
        written[grain] = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]

    profiles = {
        "daily_profile": ("EXTRACT(hour FROM timestamp_local)::INTEGER AS hour_of_day",
                          "hour_of_day"),
        "weekly_profile": ("EXTRACT(dow FROM timestamp_local)::INTEGER AS day_of_week, "
                           "EXTRACT(hour FROM timestamp_local)::INTEGER AS hour_of_day",
                           "day_of_week, hour_of_day"),
    }
    for name, (select_expr, group_expr) in profiles.items():
        path = out / "profiles" / f"{name}.parquet"
        con.execute(f"""
            COPY (
                SELECT building_id, stream_id, site_id, {select_expr},
                       COUNT(value) AS n_observations,
                       AVG(value)   AS mean_demand,
                       QUANTILE_CONT(value, 0.5) AS median_demand,
                       STDDEV_SAMP(value) AS stddev_demand
                FROM electricity
                GROUP BY building_id, stream_id, site_id, {group_expr}
                ORDER BY building_id, {group_expr}
            ) TO '{path}' (FORMAT parquet, COMPRESSION zstd)
        """)
        written[name] = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]

    con.close()
    return written
