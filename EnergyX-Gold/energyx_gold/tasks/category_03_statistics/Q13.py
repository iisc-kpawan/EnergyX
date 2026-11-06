"""Q13: Which five office buildings in the United States had the highest peak-to-average hourly electricity-consumption ratio during 2017?

Pinned definition: US office buildings with at least 90% of 2017's hours valid. A building's
peak-to-average ratio is its highest valid 2017 reading divided by its mean valid 2017 reading.
The five highest qualify; the ranking is in the evidence.

- Office is BDG2's primary space usage (`building_type_raw` = Office). US is the site's country
  (`country_canonical` = US); the anonymous sites have no country and are left out.
- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time, 8,760 hours. A building needs at least 7,884 valid hours.
- The ranking breaks ties by building_id.

    python -m energyx_gold.tasks.category_03_statistics.Q13
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_03_statistics._helpers import (
    COVERAGE,
    min_valid_hours,
    rounded,
    window_hours,
    window_params,
    year_window,
)

TASK_ID = "Q13"
COUNTRY = "US"
BUILDING_TYPE = "Office"
YEAR = 2017
TOP_N = 5

QUERY = """
SELECT b.building_id,
       MAX(e.value) / AVG(e.value) AS peak_to_average,
       MAX(e.value) AS peak_kwh,
       AVG(e.value) AS mean_kwh,
       COUNT(e.value) AS valid_hours
FROM building_sites b
LEFT JOIN electricity e
  ON e.building_id = b.building_id AND e.value IS NOT NULL
 AND e.timestamp_local >= $start AND e.timestamp_local < $end
WHERE b.country_canonical = $country AND b.building_type_raw = $building_type
GROUP BY 1
ORDER BY peak_to_average DESC NULLS LAST, b.building_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    rows = context.sql(QUERY, {"country": COUNTRY, "building_type": BUILDING_TYPE, **window_params(window)})
    needed = min_valid_hours(window)
    ranked = rows[rows["valid_hours"] >= needed].reset_index(drop=True)
    left_out = rows[rows["valid_hours"] < needed].sort_values("building_id")
    top = ranked.head(TOP_N)
    members = sorted(top["building_id"].astype(str))
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": members, "count": len(members)},
        "evidence": {
            "definition": {
                "country_canonical": COUNTRY,
                "building_type_raw": BUILDING_TYPE,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "coverage_rule": f"at least {COVERAGE:.0%} of {window_hours(window)} hours valid ({needed:g})",
                "measure": "highest valid 2017 reading / mean valid 2017 reading",
                "top_n": TOP_N,
                "ties": "broken by building_id",
                "clock": "site-local standard time",
            },
            "buildings_in_cohort": len(rows),
            "buildings_meeting_coverage": len(ranked),
            "buildings_left_out_for_coverage": {str(r.building_id): int(r.valid_hours) for r in left_out.itertuples()},
            "ranking": [
                {"rank": i + 1, "building_id": str(row.building_id), "peak_to_average": rounded(row.peak_to_average),
                 "peak_kwh": rounded(row.peak_kwh), "mean_kwh": rounded(row.mean_kwh),
                 "valid_hours": int(row.valid_hours)}
                for i, row in enumerate(ranked.itertuples())
            ],
            "lowest_qualifying_ratio": rounded(top["peak_to_average"].iloc[-1]),
            "next_ratio": rounded(ranked["peak_to_average"].iloc[TOP_N]) if len(ranked) > TOP_N else None,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
