"""Q10: How many hourly values are imputed for Peacock_office_Annie when gaps of up to six hours are filled?

Pinned definition: Peacock_office_Annie's hourly series from its first valid reading to its last.
A gap is a run of consecutive missing hours between two valid readings. Gaps of at most six hours
are filled by linear interpolation between the readings on either side; longer gaps are left
entirely unfilled. The answer is the number of filled hours. The evidence also gives the count if
longer gaps were partly filled (their first six hours).

An hour is missing when it has no row or a ZERO_MASKED row.

    python -m energyx_gold.tasks.category_02_quality.Q10
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_02_quality._helpers import inner_gaps

TASK_ID = "Q10"
BUILDING_ID = "Peacock_office_Annie"
MAX_FILL_HOURS = 6
UNIT = "count (hours)"

READINGS = """
SELECT timestamp_local AS ts, value
FROM electricity
WHERE building_id = $building_id AND value IS NOT NULL
ORDER BY ts
"""


def fill_short_gaps(series: pd.Series, max_hours: int) -> pd.Series:
    """Linearly interpolate interior gaps of at most `max_hours`; leave longer gaps missing."""
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("size")
    filled = series.interpolate(method="time", limit_area="inside")
    return filled.where(~missing | (run_length <= max_hours))


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    readings = context.sql(READINGS, {"building_id": BUILDING_ID})
    if readings.empty:
        raise ValueError(f"no valid readings for {BUILDING_ID}")
    readings["ts"] = pd.to_datetime(readings["ts"])
    hours = pd.date_range(readings["ts"].min(), readings["ts"].max(), freq="h")
    series = readings.set_index("ts")["value"].reindex(hours)
    filled = fill_short_gaps(series, MAX_FILL_HOURS)
    imputed = int((series.isna() & filled.notna()).sum())

    gaps = inner_gaps(context, building_id=BUILDING_ID)
    short = gaps[gaps["gap_hours"] <= MAX_FILL_HOURS]
    if int(short["gap_hours"].sum()) != imputed:
        raise AssertionError("interpolated hours don't match the gap lengths")
    partial = int(np.minimum(gaps["gap_hours"], MAX_FILL_HOURS).sum())
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": imputed, "unit": UNIT},
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "series": "first valid reading to last valid reading, hourly",
                "missing": "no row, or a ZERO_MASKED row",
                "max_fill_hours": MAX_FILL_HOURS,
                "method": "linear interpolation between the valid readings on either side",
                "longer_gaps": "left entirely unfilled",
            },
            "first_valid": str(readings["ts"].min()),
            "last_valid": str(readings["ts"].max()),
            "series_hours": len(hours),
            "missing_hours": int(series.isna().sum()),
            "gaps": len(gaps),
            "gaps_filled": len(short),
            "gap_length_counts": {str(k): int(v) for k, v in gaps["gap_hours"].value_counts().sort_index().items()},
            "hours_left_missing": int(filled.isna().sum()),
            "partial_fill_reading_first_six_hours_of_every_gap": partial,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
