"""Q18: How do the weekday and weekend load profiles of Cockatoo_office_Alton differ during 2017?

Pinned definition: Cockatoo_office_Alton's mean hourly profile over the weekdays of 2017 and over
the weekend days of 2017. A profile's mean hourly consumption is the mean of its 24 values. The
gap at an hour is weekday minus weekend; the largest gap is the one with the largest absolute
value. Both profiles are in the evidence.

- A profile is the mean valid reading at each hour of the day. Weekdays are Monday to Friday and
  the weekend Saturday and Sunday, in site-local standard time over 2017-01-01 00:00 up to, not
  including, 2018-01-01 00:00. A reading is valid when its value isn't null.
- The ratio is the weekday mean divided by the weekend mean.
- A peak hour is the hour with the highest mean, earliest hour first in a tie.
- The largest gap is reported with its sign (weekday minus weekend) and its hour.

    python -m energyx_gold.tasks.category_04_temporal.Q18
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_04_temporal._helpers import (
    WEEKDAY_CONDITION,
    WEEKEND_CONDITION,
    curve,
    hourly_profile,
    peak_hour,
    reading_counts,
    rounded,
    runner_up,
    year_window,
)

TASK_ID = "Q18"
BUILDING_ID = "Cockatoo_office_Alton"
YEAR = 2017


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    window = year_window(YEAR)
    weekday = hourly_profile(context, BUILDING_ID, window, WEEKDAY_CONDITION)
    weekend = hourly_profile(context, BUILDING_ID, window, WEEKEND_CONDITION)
    weekday_mean = weekday["mean_kwh"].mean()
    weekend_mean = weekend["mean_kwh"].mean()
    gap = weekday["mean_kwh"] - weekend["mean_kwh"]
    at = gap.abs().idxmax()
    gap_kwh = float(gap.loc[at])
    reading_weighted = {
        "weekday_mean_kwh": rounded((weekday["mean_kwh"] * weekday["readings"]).sum() / weekday["readings"].sum()),
        "weekend_mean_kwh": rounded((weekend["mean_kwh"] * weekend["readings"]).sum() / weekend["readings"].sum()),
    }
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": {
                "weekday_mean_kwh": rounded(weekday_mean),
                "weekend_mean_kwh": rounded(weekend_mean),
                "weekday_to_weekend_ratio": rounded(weekday_mean / weekend_mean),
                "weekday_peak_hour": peak_hour(weekday),
                "weekend_peak_hour": peak_hour(weekend),
                "largest_gap_kwh": rounded(gap_kwh),
                "largest_gap_hour": int(weekday.loc[at, "hour"]),
            },
            "units": {
                "weekday_mean_kwh": "kWh per hour",
                "weekend_mean_kwh": "kWh per hour",
                "weekday_to_weekend_ratio": "ratio",
                "weekday_peak_hour": "hour of day (local)",
                "weekend_peak_hour": "hour of day (local)",
                "largest_gap_kwh": "kWh per hour (weekday minus weekend)",
                "largest_gap_hour": "hour of day (local)",
            },
        },
        "evidence": {
            "definition": {
                "building_id": BUILDING_ID,
                "window_start_local": window[0].isoformat(),
                "window_end_local_exclusive": window[1].isoformat(),
                "weekday": "Monday to Friday (isodow 1-5)",
                "weekend": "Saturday and Sunday (isodow 6-7)",
                "profile": "mean valid kWh reading at each hour of day",
                "mean_hourly_consumption": "mean of the profile's 24 values",
                "largest_gap": "weekday minus weekend at the hour with the largest absolute difference",
                "clock": "site-local standard time",
            },
            "weekday_profile_mean_kwh_by_hour": curve(weekday),
            "weekend_profile_mean_kwh_by_hour": curve(weekend),
            "gap_kwh_by_hour": [rounded(v) for v in gap],
            "gap_direction": "weekday above weekend" if gap_kwh > 0 else "weekend above weekday",
            "weekday_readings_per_hour": reading_counts(weekday),
            "weekend_readings_per_hour": reading_counts(weekend),
            "weekday_runner_up_hour": runner_up(weekday),
            "weekend_runner_up_hour": runner_up(weekend),
            "mean_over_readings_instead_of_profile": reading_weighted,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
