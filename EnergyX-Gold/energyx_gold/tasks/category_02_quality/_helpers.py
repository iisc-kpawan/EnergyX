"""Helpers shared by the category 2 tasks (Q06 to Q10).

An hour is missing when it has no valid reading: either there is no row in `electricity`, or the
row is ZERO_MASKED, a raw zero that BDG2's cleaning removed and stored with a null value. A zero
reading is a ZERO_MASKED row. Time is site-local standard time.
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.context import GoldContext

ZERO_FLAG = "ZERO_MASKED"

# Every run of missing hours between two valid readings, per building: the hours missing between
# each valid reading and the one before it. Runs before the first valid reading and after the last
# one never appear.
INNER_GAPS_QUERY = """
WITH valid AS (
    SELECT DISTINCT e.building_id, e.timestamp_local AS ts
    FROM electricity e JOIN buildings b ON b.building_id = e.building_id
    WHERE e.value IS NOT NULL AND {filters}
),
steps AS (
    SELECT building_id, LAG(ts) OVER (PARTITION BY building_id ORDER BY ts) AS previous_ts, ts
    FROM valid
)
SELECT building_id, previous_ts, ts AS next_ts, date_diff('hour', previous_ts, ts) - 1 AS gap_hours
FROM steps
WHERE date_diff('hour', previous_ts, ts) > 1
ORDER BY building_id, previous_ts
"""

# Per building, the longest run of ZERO_MASKED rows at consecutive clock hours. A missing hour (no
# row) or a valid reading ends a run.
ZERO_RUN_CLOCK_QUERY = """
WITH zeros AS (
    SELECT DISTINCT building_id, timestamp_local AS ts
    FROM electricity
    WHERE quality_flag = $zero_flag
),
islands AS (
    SELECT building_id, ts,
           ts - to_hours(ROW_NUMBER() OVER (PARTITION BY building_id ORDER BY ts)) AS island
    FROM zeros
)
SELECT building_id, MAX(hours) AS longest_zero_run_hours
FROM (SELECT building_id, island, COUNT(*) AS hours FROM islands GROUP BY 1, 2)
GROUP BY building_id
ORDER BY building_id
"""

# Per building, the longest run of ZERO_MASKED rows over the hours that have a row. Hours with no
# row are skipped rather than breaking a run (the EXG-09 rule).
ZERO_RUN_SKIPPING_QUERY = """
WITH observed AS (
    SELECT building_id, timestamp_local AS ts, quality_flag = $zero_flag AS is_zero
    FROM electricity
),
numbered AS (
    SELECT building_id, is_zero,
           ROW_NUMBER() OVER (PARTITION BY building_id ORDER BY ts) AS seq,
           ROW_NUMBER() OVER (PARTITION BY building_id, is_zero ORDER BY ts) AS seq_in_state
    FROM observed
),
runs AS (
    SELECT building_id, seq - seq_in_state AS run_id, COUNT(*) AS hours
    FROM numbered WHERE is_zero
    GROUP BY building_id, run_id
)
SELECT building_id, MAX(hours) AS longest_zero_run_hours
FROM runs
GROUP BY building_id
ORDER BY building_id
"""


def set_result(members) -> dict:
    """The result block for a set answer: sorted unique members, and their count."""
    members = sorted({str(member) for member in members})
    return {"answer": members, "count": len(members)}


def inner_gaps(context: GoldContext, building_id: str | None = None,
               building_type: str | None = None) -> pd.DataFrame:
    """Every run of missing hours between two valid readings, for one building or one type."""
    filters, params = ["TRUE"], {}
    if building_id is not None:
        filters.append("e.building_id = $building_id")
        params["building_id"] = building_id
    if building_type is not None:
        filters.append("b.building_type_raw = $building_type")
        params["building_type"] = building_type
    frame = context.sql(INNER_GAPS_QUERY.format(filters=" AND ".join(filters)), params)
    frame["gap_hours"] = frame["gap_hours"].astype(int)
    return frame


def zero_runs(context: GoldContext, skip_missing: bool) -> pd.DataFrame:
    """Per building with a zero reading, its longest run of zero readings."""
    query = ZERO_RUN_SKIPPING_QUERY if skip_missing else ZERO_RUN_CLOCK_QUERY
    frame = context.sql(query, {"zero_flag": ZERO_FLAG})
    frame["longest_zero_run_hours"] = frame["longest_zero_run_hours"].astype(int)
    return frame
