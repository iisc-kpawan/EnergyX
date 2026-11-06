"""Q44: How many times did the daily electricity consumption of Cockatoo_office_Alton increase by more than 20% compared with the previous day during 2016-2017?

Pinned definition: Cockatoo_office_Alton's daily totals over 2016-2017, counting only days with 24
valid readings. An increase is a day whose total is more than 1.2 times the previous calendar
day's total, when both days count. The dates are in the evidence.

- Days are local calendar days, 1 January 2016 to 31 December 2017.
- "More than" is strict: total > 1.2 x previous total.
- The first day, 1 January 2016, has no previous day in the period.

    python -m energyx_gold.tasks.category_09_anomaly.Q44
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_09_anomaly import _helpers as helpers

TASK_ID = "Q44"
BUILDING = "Cockatoo_office_Alton"
START = "2016-01-01"
END = "2018-01-01"  # exclusive
RATIO = 1.2
HOURS_PER_DAY = 24
UNIT = "count (days)"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    readings = helpers.valid_readings(context, [BUILDING], START, END)
    day = readings["timestamp_local"].dt.normalize()
    daily = readings.groupby(day)["value"].agg(["sum", "count"])
    days = pd.date_range(START, END, freq="D", inclusive="left")
    daily = daily.reindex(days)
    complete = daily["count"] == HOURS_PER_DAY
    total = daily["sum"].where(complete)
    previous = total.shift(1)
    increase = total > RATIO * previous  # False where either day doesn't count
    both = total.notna() & previous.notna()
    ratios = total[both] / previous[both]
    dates = [d.strftime("%Y-%m-%d") for d in total.index[increase]]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": len(dates), "unit": UNIT},
        "evidence": {
            "definition": {
                "building": BUILDING,
                "period": f"{START} to 2017-12-31, site-local calendar days",
                "complete_day": f"{HOURS_PER_DAY} valid readings",
                "increase": f"total > {RATIO} x previous calendar day's total, both days complete",
            },
            "days_in_period": int(len(days)),
            "complete_days": int(complete.sum()),
            "day_pairs_compared": int(both.sum()),
            "increase_dates": dates,
            "increase_ratios": {d.strftime("%Y-%m-%d"): helpers.rounded(total[d] / previous[d])
                                for d in total.index[increase]},
            "ratio_nearest_threshold": helpers.rounded(ratios.loc[(ratios - RATIO).abs().idxmin()]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
