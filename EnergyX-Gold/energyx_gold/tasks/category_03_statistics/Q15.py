"""Q15: How did the total annual electricity consumption of the largest 10% of buildings by floor area compare with that of the smallest 50% in 2017?

Pinned definition: buildings with 2017 readings are ranked by floor area. The largest 10% are
those at or above the 90th percentile of floor area, and the smallest 50% those at or below the
median. Each group's total is the sum of its buildings' valid 2017 readings.

- A building has 2017 readings when it has at least one valid reading in 2017. There is no
  coverage rule, since the question asks for totals.
- Percentiles are over the floor areas of those buildings, with linear interpolation between
  neighbouring values (QUANTILE_CONT, numpy's default).
- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time. A reading is valid when its value isn't null.
- Every such building has a floor area; the task fails if one does not.
- The ratio is the largest group's total divided by the smallest group's total.

    python -m energyx_gold.tasks.category_03_statistics.Q15
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_03_statistics._helpers import rounded, window_params, year_window

TASK_ID = "Q15"
YEAR = 2017
LARGEST_QUANTILE = 0.9
SMALLEST_QUANTILE = 0.5

QUERY = """
WITH totals AS (
    SELECT e.building_id, b.floor_area_m2, SUM(e.value) AS kwh, COUNT(e.value) AS valid_hours
    FROM electricity e
    JOIN buildings b USING (building_id)
    WHERE e.value IS NOT NULL
      AND e.timestamp_local >= $start AND e.timestamp_local < $end
    GROUP BY 1, 2
),
cuts AS (
    SELECT QUANTILE_CONT(floor_area_m2, $largest_q) AS largest_cut_m2,
           QUANTILE_CONT(floor_area_m2, $smallest_q) AS smallest_cut_m2,
           COUNT(*) AS buildings,
           COUNT(floor_area_m2) FILTER (WHERE floor_area_m2 > 0) AS with_floor_area
    FROM totals
)
SELECT c.largest_cut_m2, c.smallest_cut_m2, c.buildings, c.with_floor_area,
       COUNT(*) FILTER (WHERE t.floor_area_m2 >= c.largest_cut_m2) AS largest_buildings,
       SUM(t.kwh) FILTER (WHERE t.floor_area_m2 >= c.largest_cut_m2) AS largest_kwh,
       MIN(t.floor_area_m2) FILTER (WHERE t.floor_area_m2 >= c.largest_cut_m2) AS largest_min_m2,
       COUNT(*) FILTER (WHERE t.floor_area_m2 <= c.smallest_cut_m2) AS smallest_buildings,
       SUM(t.kwh) FILTER (WHERE t.floor_area_m2 <= c.smallest_cut_m2) AS smallest_kwh,
       MAX(t.floor_area_m2) FILTER (WHERE t.floor_area_m2 <= c.smallest_cut_m2) AS smallest_max_m2,
       SUM(t.kwh) AS all_kwh
FROM totals t, cuts c
GROUP BY ALL
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    row = context.sql(QUERY, {"largest_q": LARGEST_QUANTILE, "smallest_q": SMALLEST_QUANTILE,
                              **window_params(window)}).iloc[0]
    if int(row["with_floor_area"]) != int(row["buildings"]):
        raise ValueError(f"{TASK_ID}: some buildings with 2017 readings have no valid floor area")
    ratio = row["largest_kwh"] / row["smallest_kwh"]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": {
                "largest_10pct_total_kwh": rounded(row["largest_kwh"]),
                "smallest_50pct_total_kwh": rounded(row["smallest_kwh"]),
                "largest_10pct_buildings": int(row["largest_buildings"]),
                "smallest_50pct_buildings": int(row["smallest_buildings"]),
                "ratio_largest_to_smallest": rounded(ratio),
            },
            "units": {
                "largest_10pct_total_kwh": "kWh",
                "smallest_50pct_total_kwh": "kWh",
                "largest_10pct_buildings": "count (buildings)",
                "smallest_50pct_buildings": "count (buildings)",
                "ratio_largest_to_smallest": "ratio",
            },
        },
        "evidence": {
            "definition": {
                "cohort": "buildings with at least one valid 2017 reading",
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "largest_10pct": f"floor_area_m2 >= quantile_cont({LARGEST_QUANTILE})",
                "smallest_50pct": f"floor_area_m2 <= quantile_cont({SMALLEST_QUANTILE})",
                "total": "sum of a group's buildings' valid 2017 readings",
                "clock": "site-local standard time",
            },
            "buildings_with_2017_readings": int(row["buildings"]),
            "largest_cut_m2": rounded(row["largest_cut_m2"]),
            "smallest_cut_m2": rounded(row["smallest_cut_m2"]),
            "largest_group_min_floor_area_m2": rounded(row["largest_min_m2"]),
            "smallest_group_max_floor_area_m2": rounded(row["smallest_max_m2"]),
            "all_buildings_total_kwh": rounded(row["all_kwh"]),
            "largest_share_of_all_kwh": rounded(row["largest_kwh"] / row["all_kwh"]),
            "smallest_share_of_all_kwh": rounded(row["smallest_kwh"] / row["all_kwh"]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
