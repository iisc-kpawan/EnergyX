"""Q16: At what hour of the day did Cockatoo_office_Alton have its highest average hourly electricity consumption in 2017?

Pinned definition: Cockatoo_office_Alton's mean valid reading for each hour of the day over 2017.
The answer is the hour with the highest mean.

- The year runs from 2017-01-01 00:00 up to, not including, 2018-01-01 00:00 in site-local
  standard time. A reading is valid when its value isn't null.
- The hour is the hour the reading starts, 0 to 23. The earliest hour wins a tie.

    python -m energyx_gold.tasks.category_04_temporal.Q16
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_04_temporal._helpers import (
    curve,
    hourly_profile,
    peak_hour,
    reading_counts,
    rounded,
    runner_up,
    year_window,
)

TASK_ID = "Q16"
BUILDING_ID = "Cockatoo_office_Alton"
YEAR = 2017
UNIT = "hour of day (local)"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    profile = hourly_profile(context, BUILDING_ID, window)
    hour = peak_hour(profile)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": hour, "unit": UNIT},
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "profile": "mean valid kWh reading at each hour of day",
                "ties": "earliest hour",
                "clock": "site-local standard time",
            },
            "peak_mean_kwh": rounded(profile.loc[profile["hour"] == hour, "mean_kwh"].iloc[0]),
            "runner_up_hour": runner_up(profile),
            "profile_mean_kwh_by_hour": curve(profile),
            "readings_per_hour": reading_counts(profile),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
