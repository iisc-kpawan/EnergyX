"""Helpers shared by the category 7 questions (Q31 to Q35).

Conventions from data/gold50_questions.yaml:
- a reading is valid when its value isn't null, so masked zeros and absent hours are both missing;
- a building qualifies for a yearly measure when at least 90% of the year's hours are valid;
- EUI is the year's valid kWh divided by floor_area_m2;
- a building type is BDG2's primaryspaceusage, held as building_type_raw;
- city and country are the site's (sites.city, sites.country_canonical). The five anonymous sites
  have neither;
- weekdays are Monday to Friday; night is hours 20 to 23 and 0 to 6, day is hours 7 to 19.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from energyx_gold.analytics import clustering
from energyx_gold.core.context import GoldContext

DECIMALS = 6
MIN_COVERAGE = 0.90
OFFICE = "Office"
EDUCATION = "Education"
NIGHT_HOURS = (20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6)
DAY_HOURS = tuple(range(7, 20))
HOURS = list(range(24))

# One row per building with a valid reading in the window, with its site's city and country.
ANNUAL_QUERY = """
SELECT b.building_id,
       b.site_id,
       b.building_type_raw AS building_type,
       s.city,
       s.country_canonical AS country,
       b.floor_area_m2,
       COUNT(e.value) AS valid_hours,
       SUM(e.value) AS total_kwh
FROM buildings b
JOIN sites s ON s.site_id = b.site_id
JOIN electricity e ON e.building_id = b.building_id
WHERE e.timestamp_local >= $start AND e.timestamp_local < $end AND e.value IS NOT NULL
GROUP BY b.building_id, b.site_id, b.building_type_raw, s.city, s.country_canonical, b.floor_area_m2
ORDER BY b.building_id
"""

# Mean valid weekday reading by hour of the day, per building of one type, inside the window.
WEEKDAY_PROFILE_QUERY = """
SELECT e.building_id,
       EXTRACT(hour FROM e.timestamp_local) AS hour,
       AVG(e.value) AS mean_kwh
FROM electricity e
JOIN buildings b ON b.building_id = e.building_id
WHERE b.building_type_raw = $building_type
  AND e.value IS NOT NULL
  AND ISODOW(e.timestamp_local) <= 5
  AND e.timestamp_local >= $start AND e.timestamp_local < $end
GROUP BY e.building_id, EXTRACT(hour FROM e.timestamp_local)
"""


def rounded(value) -> float:
    return round(float(value), DECIMALS)


def year_window(year: int) -> tuple[datetime, datetime]:
    """Start (inclusive) and end (exclusive) of a local calendar year."""
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def hours_in_year(year: int) -> int:
    start, end = year_window(year)
    return int((end - start).total_seconds() // 3600)


def annual_by_building(context: GoldContext, year: int) -> pd.DataFrame:
    """Valid hours, kWh, coverage and EUI per building with a valid reading in `year`."""
    start, end = year_window(year)
    frame = context.sql(ANNUAL_QUERY, {"start": start, "end": end}).set_index("building_id")
    frame["coverage"] = frame["valid_hours"] / hours_in_year(year)
    area = frame["floor_area_m2"].where(frame["floor_area_m2"] > 0)
    frame["eui"] = frame["total_kwh"] / area
    return frame


def qualifying(frame: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    return frame[frame["coverage"] >= min_coverage]


def of_type(frame: pd.DataFrame, building_type: str) -> pd.DataFrame:
    return frame[frame["building_type"] == building_type]


def qualifying_office_profiles(context: GoldContext, year: int) -> tuple[pd.DataFrame, dict]:
    """Normalized weekday profiles of the office buildings with at least 90% of the year valid.

    Returns one row per building: columns 0 to 23 hold the profile scaled to 0-1 by its own
    minimum and maximum (a flat profile becomes all zeros), plus site_id, city and country.
    Also returns counts a reviewer wants: qualifying offices, and those left out for a missing
    hour of the day.
    """
    annual = annual_by_building(context, year)
    offices = of_type(qualifying(annual), OFFICE)
    start, end = year_window(year)
    rows = context.sql(WEEKDAY_PROFILE_QUERY, {"building_type": OFFICE, "start": start, "end": end})
    rows["hour"] = rows["hour"].astype(int)
    wide = rows.pivot(index="building_id", columns="hour", values="mean_kwh").reindex(columns=HOURS)
    wide = wide.reindex(offices.index)
    complete = wide[wide.notna().all(axis=1)]
    scaled = clustering.minmax_rows(complete)
    scaled = scaled.join(offices[["site_id", "city", "country"]]).sort_index()
    flat = int((complete.max(axis=1) == complete.min(axis=1)).sum())
    info = {
        "qualifying_offices": int(len(offices)),
        "left_out_missing_an_hour": sorted(set(offices.index) - set(complete.index)),
        "flat_profiles": flat,
    }
    return scaled, info


def group_profile(scaled: pd.DataFrame, by: str) -> pd.DataFrame:
    """The mean scaled profile of each group, one row per group and one column per hour."""
    return scaled.groupby(by)[HOURS].mean()


def night_to_day(profile: pd.Series) -> float:
    return float(profile[list(NIGHT_HOURS)].mean() / profile[list(DAY_HOURS)].mean())
