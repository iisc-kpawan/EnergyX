"""Q11: What was the total electricity consumption of Rat_office_Adele in July 2016?

Pinned definition: the sum of Rat_office_Adele's valid hourly readings in July 2016.

- The window is 2016-07-01 00:00 up to, not including, 2016-08-01 00:00 in site-local standard
  time, which is 744 hours.
- A reading is valid when its value isn't null. A missing hour adds nothing to the total.

    python -m energyx_gold.tasks.category_03_statistics.Q11
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_03_statistics._helpers import month_window, rounded, window_hours, window_params

TASK_ID = "Q11"
BUILDING = "Rat_office_Adele"
YEAR, MONTH = 2016, 7
UNIT = "kWh"

QUERY = """
SELECT SUM(value) AS total_kwh,
       COUNT(value) AS valid_hours,
       COUNT(*) AS rows_present,
       MIN(value) AS min_kwh,
       MAX(value) AS max_kwh,
       AVG(value) AS mean_kwh
FROM electricity
WHERE building_id = $building
  AND timestamp_local >= $start AND timestamp_local < $end
"""


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = month_window(YEAR, MONTH)
    row = context.sql(QUERY, {"building": BUILDING, **window_params(window)}).iloc[0]
    hours = window_hours(window)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": rounded(row["total_kwh"]), "unit": UNIT},
        "evidence": {
            "definition": {
                "building_id": BUILDING,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "measure": "sum of valid hourly kWh readings",
                "clock": "site-local standard time",
            },
            "window_hours": hours,
            "valid_hours": int(row["valid_hours"]),
            "missing_hours": hours - int(row["valid_hours"]),
            "masked_zero_rows": int(row["rows_present"] - row["valid_hours"]),
            "min_kwh": rounded(row["min_kwh"]),
            "max_kwh": rounded(row["max_kwh"]),
            "mean_kwh": rounded(row["mean_kwh"]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
