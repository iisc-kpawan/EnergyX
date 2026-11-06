"""Helpers shared by the category 4 tasks: hourly load profiles in site-local standard time.

A profile is the mean of a building's valid readings at each hour of the day. A reading is valid
when its value isn't null, so hours with no row and ZERO_MASKED rows both drop out. Readings are
kWh per hour.

Weekdays are Monday to Friday and the weekend Saturday and Sunday, by the local timestamp's ISO
weekday.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from energyx_gold.core.context import GoldContext

HOURS = list(range(24))
DECIMALS = 6
COVERAGE = 0.9
WEEKDAY_CONDITION = "AND isodow(timestamp_local) <= 5"
WEEKEND_CONDITION = "AND isodow(timestamp_local) >= 6"

PROFILE_QUERY = """
SELECT hour(timestamp_local) AS hour, AVG(value) AS mean_kwh, COUNT(value) AS readings
FROM electricity
WHERE building_id = $building_id AND value IS NOT NULL
  AND timestamp_local >= $start AND timestamp_local < $end {condition}
GROUP BY 1
ORDER BY 1
"""


def rounded(value) -> float:
    """A float rounded to the result precision."""
    return round(float(value), DECIMALS)


def year_window(year: int) -> tuple[datetime, datetime]:
    """Start (inclusive) and end (exclusive) of a local calendar year."""
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def window_params(window: tuple[datetime, datetime]) -> dict:
    """The start and end of a window as query parameters."""
    return {"start": window[0], "end": window[1]}


def window_hours(window: tuple[datetime, datetime]) -> int:
    """The number of hours in a window."""
    return int((window[1] - window[0]).total_seconds() // 3600)


def hourly_profile(context: GoldContext, building_id: str, window: tuple[datetime, datetime],
                   condition: str = "") -> pd.DataFrame:
    """Mean kWh at each hour 0-23 inside `window`. `condition` is an extra SQL clause starting with AND."""
    profile = context.sql(PROFILE_QUERY.format(condition=condition),
                          {"building_id": building_id, **window_params(window)})
    if profile["hour"].tolist() != HOURS:
        raise ValueError(f"{building_id}: profile doesn't cover all 24 hours")
    return profile


def curve(profile: pd.DataFrame, column: str = "mean_kwh") -> list[float]:
    """The 24 values of a profile, for evidence."""
    return [rounded(v) for v in profile[column]]


def reading_counts(profile: pd.DataFrame) -> dict:
    """The fewest and most readings behind any hour of a profile."""
    return {"min": int(profile["readings"].min()), "max": int(profile["readings"].max()),
            "total": int(profile["readings"].sum())}


def peak_hour(profile: pd.DataFrame) -> int:
    """The hour with the highest mean. The earliest hour wins a tie."""
    return int(profile.loc[profile["mean_kwh"].idxmax(), "hour"])


def runner_up(profile: pd.DataFrame) -> dict:
    """The second-highest hour and its mean, to show how close the peak is."""
    second = profile.sort_values(["mean_kwh", "hour"], ascending=[False, True]).iloc[1]
    return {"hour": int(second["hour"]), "mean_kwh": rounded(second["mean_kwh"])}
