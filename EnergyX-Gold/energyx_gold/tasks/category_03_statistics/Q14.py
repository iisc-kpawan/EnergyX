"""Q14: Which building types in London had at least one month in 2017 when average daily electricity consumption was more than 20% above their annual average daily consumption?

Pinned definition: London buildings with at least 90% of 2017's hours valid, grouped by
primaryspaceusage. A type's daily consumption is the sum of its buildings' valid readings on that
day. Its monthly average daily consumption is the month's total divided by the month's days, and
its annual average daily consumption is the 2017 total divided by 365. A type qualifies when some
month's average is more than 1.2 times the annual average. Every type's monthly ratios are in the
evidence.

- London is the site's city (the Mouse, Robin and Shrew sites). The type is BDG2's
  primaryspaceusage (`building_type_raw`); buildings with no label are left out.
- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time, 8,760 hours. A building needs at least 7,884 valid hours.
- A month's ratio is (month total / days in month) / (year total / 365). Missing hours add nothing.

    python -m energyx_gold.tasks.category_03_statistics.Q14
"""

from __future__ import annotations

import calendar

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_03_statistics._helpers import (
    COVERAGE,
    min_valid_hours,
    rounded,
    window_hours,
    window_params,
    year_window,
)

TASK_ID = "Q14"
CITY = "London"
YEAR = 2017
DAYS_IN_YEAR = 365
THRESHOLD = 1.2

COVERAGE_QUERY = """
SELECT b.building_id, b.site_id, b.building_type_raw, COUNT(e.value) AS valid_hours
FROM building_sites b
LEFT JOIN electricity e
  ON e.building_id = b.building_id AND e.value IS NOT NULL
 AND e.timestamp_local >= $start AND e.timestamp_local < $end
WHERE b.city = $city
GROUP BY ALL
ORDER BY b.building_id
"""

MONTHLY_QUERY = """
SELECT b.building_type_raw AS building_type, month(e.timestamp_local) AS month, SUM(e.value) AS total_kwh
FROM electricity e
JOIN buildings b USING (building_id)
WHERE e.building_id IN (SELECT UNNEST($buildings)) AND e.value IS NOT NULL
  AND e.timestamp_local >= $start AND e.timestamp_local < $end
GROUP BY 1, 2
ORDER BY 1, 2
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    buildings = context.sql(COVERAGE_QUERY, {"city": CITY, **window_params(window)})
    needed = min_valid_hours(window)
    labelled = buildings[buildings["building_type_raw"].notna()]
    cohort = labelled[labelled["valid_hours"] >= needed]
    monthly = context.sql(MONTHLY_QUERY, {"buildings": sorted(cohort["building_id"].astype(str)),
                                          **window_params(window)})
    types = {}
    for building_type, part in monthly.groupby("building_type", sort=True):
        totals = dict(zip(part["month"].astype(int), part["total_kwh"]))
        annual_daily = sum(totals.values()) / DAYS_IN_YEAR
        ratios = {m: totals.get(m, 0.0) / calendar.monthrange(YEAR, m)[1] / annual_daily for m in range(1, 13)}
        peak_month = max(ratios, key=lambda m: (ratios[m], -m))
        types[str(building_type)] = {
            "buildings": int((cohort["building_type_raw"] == building_type).sum()),
            "annual_average_daily_kwh": rounded(annual_daily),
            "monthly_ratio": {calendar.month_name[m]: rounded(r) for m, r in ratios.items()},
            "highest_month": calendar.month_name[peak_month],
            "highest_ratio": rounded(ratios[peak_month]),
            "qualifies": bool(ratios[peak_month] > THRESHOLD),
        }
    members = sorted(t for t, v in types.items() if v["qualifies"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": members, "count": len(members)},
        "evidence": {
            "definition": {
                "city": CITY,
                "type": "building_type_raw (BDG2 primaryspaceusage)",
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "coverage_rule": f"at least {COVERAGE:.0%} of {window_hours(window)} hours valid ({needed:g})",
                "monthly_ratio": "(month total / days in month) / (2017 total / 365)",
                "threshold": f"some month's ratio > {THRESHOLD}",
                "clock": "site-local standard time",
            },
            "london_buildings": len(buildings),
            "buildings_without_type": sorted(buildings.loc[buildings["building_type_raw"].isna(), "building_id"]),
            "buildings_left_out_for_coverage": {str(r.building_id): int(r.valid_hours)
                                                for r in labelled[labelled["valid_hours"] < needed].itertuples()},
            "buildings_in_cohort": len(cohort),
            "types": types,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
