"""Q20: Which five education buildings in London showed the biggest differences between their weekday and weekend hourly load profiles during 2017?

Pinned definition: London education buildings with at least 90% of 2017's hours valid. Each
building's weekday and weekend mean hourly profiles for 2017 are each normalized to 0-1. The
difference is the mean absolute gap between the two normalized profiles over the 24 hours. The
five largest qualify; the ranking is in the evidence.

- London is the site's city (the Mouse, Robin and Shrew sites). Education is BDG2's
  primaryspaceusage (`building_type_raw`).
- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time, 8,760 hours; a building needs at least 7,884 valid hours. A reading is valid
  when its value isn't null.
- Weekdays are Monday to Friday and the weekend Saturday and Sunday. A profile is the mean valid
  reading at each hour of the day, scaled to 0-1 by its own minimum and maximum.
- The ranking breaks ties by building_id.

    python -m energyx_gold.tasks.category_04_temporal.Q20
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.analytics.clustering import minmax_rows
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_04_temporal._helpers import (
    COVERAGE,
    HOURS,
    rounded,
    window_hours,
    window_params,
    year_window,
)

TASK_ID = "Q20"
CITY = "London"
BUILDING_TYPE = "Education"
YEAR = 2017
TOP_N = 5

COVERAGE_QUERY = """
SELECT b.building_id, COUNT(e.value) AS valid_hours
FROM building_sites b
LEFT JOIN electricity e
  ON e.building_id = b.building_id AND e.value IS NOT NULL
 AND e.timestamp_local >= $start AND e.timestamp_local < $end
WHERE b.city = $city AND b.building_type_raw = $building_type
GROUP BY 1
ORDER BY 1
"""

PROFILE_QUERY = """
SELECT building_id,
       isodow(timestamp_local) <= 5 AS weekday,
       hour(timestamp_local) AS hour,
       AVG(value) AS mean_kwh,
       COUNT(value) AS readings
FROM electricity
WHERE building_id IN (SELECT UNNEST($buildings)) AND value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end
GROUP BY 1, 2, 3
"""


def profile_matrix(rows: pd.DataFrame, weekday: bool) -> pd.DataFrame:
    """One row per building, one column per hour of day, for weekdays or for the weekend."""
    part = rows[rows["weekday"] == weekday]
    return part.pivot(index="building_id", columns="hour", values="mean_kwh").reindex(columns=HOURS)


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    coverage = context.sql(COVERAGE_QUERY, {"city": CITY, "building_type": BUILDING_TYPE, **window_params(window)})
    needed = COVERAGE * window_hours(window)
    buildings = sorted(coverage.loc[coverage["valid_hours"] >= needed, "building_id"].astype(str))
    rows = context.sql(PROFILE_QUERY, {"buildings": buildings, **window_params(window)})
    weekday = profile_matrix(rows, True).reindex(index=buildings)
    weekend = profile_matrix(rows, False).reindex(index=buildings)
    if weekday.isna().any().any() or weekend.isna().any().any():
        raise ValueError(f"{TASK_ID}: a building's profile doesn't cover all 24 hours")
    scaled_weekday = minmax_rows(weekday)
    scaled_weekend = minmax_rows(weekend)
    gap = (scaled_weekday - scaled_weekend).abs()
    ranked = (pd.DataFrame({"mean_abs_gap": gap.mean(axis=1)})
              .rename_axis("building_id").reset_index()
              .sort_values(["mean_abs_gap", "building_id"], ascending=[False, True], kind="stable")
              .reset_index(drop=True))
    top = ranked.head(TOP_N)
    members = sorted(top["building_id"].astype(str))
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": members, "count": len(members)},
        "evidence": {
            "definition": {
                "city": CITY,
                "building_type_raw": BUILDING_TYPE,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "coverage_rule": f"at least {COVERAGE:.0%} of {window_hours(window)} hours valid ({needed:g})",
                "weekday": "Monday to Friday (isodow 1-5)",
                "weekend": "Saturday and Sunday (isodow 6-7)",
                "profile": "mean valid kWh reading at each hour of day",
                "scaling": "each profile to 0-1 by its own minimum and maximum",
                "difference": "mean over 24 hours of the absolute gap between the scaled profiles",
                "top_n": TOP_N,
                "ties": "broken by building_id",
                "clock": "site-local standard time",
            },
            "london_education_buildings": len(coverage),
            "buildings_left_out_for_coverage": {str(r.building_id): int(r.valid_hours)
                                                for r in coverage[coverage["valid_hours"] < needed].itertuples()},
            "buildings_ranked": len(ranked),
            "ranking": [
                {"rank": i + 1, "building_id": str(r.building_id), "mean_abs_gap": rounded(r.mean_abs_gap)}
                for i, r in enumerate(ranked.itertuples())
            ],
            "lowest_qualifying_gap": rounded(top["mean_abs_gap"].iloc[-1]),
            "next_gap": rounded(ranked["mean_abs_gap"].iloc[TOP_N]) if len(ranked) > TOP_N else None,
            "scaled_profiles_of_members": {
                building: {
                    "weekday": [rounded(v) for v in scaled_weekday.loc[building]],
                    "weekend": [rounded(v) for v in scaled_weekend.loc[building]],
                }
                for building in members
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
