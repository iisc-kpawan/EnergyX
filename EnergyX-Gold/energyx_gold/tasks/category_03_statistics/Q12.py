"""Q12: On which day did Fox_office_Alice have the highest total electricity consumption in 2017, and what was the total on that day?

Pinned definition: Fox_office_Alice's daily total is the sum of its valid readings on each local
calendar day of 2017. The answer is the day with the highest total and that total. The date must
match exactly; the evidence gives the day's number of valid hours and the runner-up day.

- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time. A reading is valid when its value isn't null.
- A day with no valid reading has no total. A tie goes to the earlier day.

    python -m energyx_gold.tasks.category_03_statistics.Q12
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_03_statistics._helpers import rounded, window_params, year_window

TASK_ID = "Q12"
BUILDING = "Fox_office_Alice"
YEAR = 2017
TOP_SHOWN = 5

QUERY = """
SELECT CAST(timestamp_local AS DATE) AS day,
       SUM(value) AS total_kwh,
       COUNT(value) AS valid_hours
FROM electricity
WHERE building_id = $building AND value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end
GROUP BY 1
ORDER BY total_kwh DESC, day
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    days = context.sql(QUERY, {"building": BUILDING, **window_params(window)})
    top = days.iloc[0]
    second = days.iloc[1]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": {"date": top["day"].strftime("%Y-%m-%d"), "total_kwh": rounded(top["total_kwh"])},
            "units": {"date": "local date (YYYY-MM-DD)", "total_kwh": "kWh"},
        },
        "evidence": {
            "definition": {
                "building_id": BUILDING,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "daily_total": "sum of valid readings on the local calendar day",
                "ties": "earlier day first",
                "clock": "site-local standard time",
            },
            "peak_day_valid_hours": int(top["valid_hours"]),
            "runner_up": {"date": second["day"].strftime("%Y-%m-%d"), "total_kwh": rounded(second["total_kwh"]),
                          "valid_hours": int(second["valid_hours"])},
            "days_with_readings": len(days),
            "days_with_24_valid_hours": int((days["valid_hours"] == 24).sum()),
            "top_days": [
                {"date": row.day.strftime("%Y-%m-%d"), "total_kwh": rounded(row.total_kwh),
                 "valid_hours": int(row.valid_hours)}
                for row in days.head(TOP_SHOWN).itertuples()
            ],
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
