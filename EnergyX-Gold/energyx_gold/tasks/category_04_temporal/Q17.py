"""Q17: During which month did Bear_education_Alfredo have the highest total electricity consumption in 2017?

Pinned definition: Bear_education_Alfredo's monthly total is the sum of its valid readings in each
month of 2017. The answer is the month with the highest total; every month's total and valid-hour
count is in the evidence.

- Months are local calendar months of 2017 in site-local standard time. A reading is valid when
  its value isn't null; a missing hour adds nothing.
- The answer is the English month name. The earlier month wins a tie.

    python -m energyx_gold.tasks.category_04_temporal.Q17
"""

from __future__ import annotations

import calendar

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_04_temporal._helpers import rounded, window_params, year_window

TASK_ID = "Q17"
BUILDING_ID = "Bear_education_Alfredo"
YEAR = 2017

QUERY = """
SELECT month(timestamp_local) AS month, SUM(value) AS total_kwh, COUNT(value) AS valid_hours
FROM electricity
WHERE building_id = $building_id AND value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end
GROUP BY 1
ORDER BY 1
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    months = context.sql(QUERY, {"building_id": BUILDING_ID, **window_params(window)})
    ranked = months.sort_values(["total_kwh", "month"], ascending=[False, True]).reset_index(drop=True)
    top, second = ranked.iloc[0], ranked.iloc[1]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": calendar.month_name[int(top["month"])]},
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "monthly_total": "sum of valid kWh readings in the local calendar month",
                "ties": "earlier month",
                "clock": "site-local standard time",
            },
            "months": [
                {"month": calendar.month_name[int(r.month)], "total_kwh": rounded(r.total_kwh),
                 "valid_hours": int(r.valid_hours),
                 "hours_in_month": calendar.monthrange(YEAR, int(r.month))[1] * 24}
                for r in months.itertuples()
            ],
            "peak_total_kwh": rounded(top["total_kwh"]),
            "runner_up": {"month": calendar.month_name[int(second["month"])], "total_kwh": rounded(second["total_kwh"])},
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
