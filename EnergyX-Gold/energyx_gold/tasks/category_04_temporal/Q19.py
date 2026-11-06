"""Q19: Which five buildings in Berkeley, CA had the highest average electricity consumption between 2 PM and 6 PM during June-August 2017?

Pinned definition: buildings at the Berkeley site (Bear). The afternoon window is the hours
starting 14:00, 15:00, 16:00 and 17:00 on every day from 1 June to 31 August 2017, 368 hours in
all. Buildings with valid readings for at least 90% of those hours count, and their value is the
mean of those readings. The five highest qualify; the ranking is in the evidence.

- Berkeley, CA is `site_id` Bear. Times are site-local standard time. A reading is valid when its
  value isn't null. 90% of 368 hours is 331.2, so a building needs at least 332 valid hours.
- The ranking breaks ties by building_id.

    python -m energyx_gold.tasks.category_04_temporal.Q19
"""

from __future__ import annotations

from datetime import datetime

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_04_temporal._helpers import COVERAGE, rounded

TASK_ID = "Q19"
SITE = "Bear"
START, END = datetime(2017, 6, 1), datetime(2017, 9, 1)
FIRST_HOUR, LAST_HOUR = 14, 17  # hours starting 14:00 to 17:00, so 14:00-17:59
WINDOW_HOURS = 92 * 4
TOP_N = 5

QUERY = """
SELECT b.building_id,
       AVG(e.value) AS mean_kwh,
       COUNT(e.value) AS valid_hours
FROM buildings b
LEFT JOIN electricity e
  ON e.building_id = b.building_id AND e.value IS NOT NULL
 AND e.timestamp_local >= $start AND e.timestamp_local < $end
 AND hour(e.timestamp_local) BETWEEN $first_hour AND $last_hour
WHERE b.site_id = $site
GROUP BY 1
ORDER BY mean_kwh DESC NULLS LAST, b.building_id
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = context.sql(QUERY, {"site": SITE, "start": START, "end": END,
                               "first_hour": FIRST_HOUR, "last_hour": LAST_HOUR})
    needed = COVERAGE * WINDOW_HOURS
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
                "site_id": SITE,
                "window_local": f"{START.date()} to {(END.date())} exclusive, hours {FIRST_HOUR}:00 to {LAST_HOUR}:00 start",
                "window_hours": WINDOW_HOURS,
                "coverage_rule": f"at least {COVERAGE:.0%} of {WINDOW_HOURS} hours valid ({needed:g})",
                "measure": "mean of valid readings in the window",
                "top_n": TOP_N,
                "ties": "broken by building_id",
                "clock": "site-local standard time",
            },
            "buildings_at_site": len(rows),
            "buildings_meeting_coverage": len(ranked),
            "buildings_left_out_for_coverage": {str(r.building_id): int(r.valid_hours) for r in left_out.itertuples()},
            "ranking": [
                {"rank": i + 1, "building_id": str(r.building_id), "mean_kwh": rounded(r.mean_kwh),
                 "valid_hours": int(r.valid_hours)}
                for i, r in enumerate(ranked.itertuples())
            ],
            "lowest_qualifying_mean_kwh": rounded(top["mean_kwh"].iloc[-1]),
            "next_mean_kwh": rounded(ranked["mean_kwh"].iloc[TOP_N]) if len(ranked) > TOP_N else None,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
