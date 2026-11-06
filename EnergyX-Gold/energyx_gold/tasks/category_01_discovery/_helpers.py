"""Helpers shared by the category 1 tasks (Q01 to Q05).

A reading is valid when its value isn't null, so ZERO_MASKED rows don't count. Time is site-local
standard time, so 2016 has 8,784 hours and 2017 has 8,760. Valid floor-area data means
floor_area_m2 is present and greater than 0.
"""

from __future__ import annotations

from datetime import datetime, timedelta

YEARS = (2016, 2017)
FLOOR_AREA_FILTER = "b.floor_area_m2 IS NOT NULL AND b.floor_area_m2 > 0"


def hours_between(start: datetime, end: datetime) -> int:
    """Hours from `start` (inclusive) to `end` (exclusive)."""
    return (end - start) // timedelta(hours=1)


def year_hours(year: int) -> int:
    """Hours in a local calendar year."""
    return hours_between(datetime(year, 1, 1), datetime(year + 1, 1, 1))


def month_start(year: int, month: int) -> datetime:
    """The first hour of a month; month 13 rolls over to January of the next year."""
    return datetime(year + (month - 1) // 12, (month - 1) % 12 + 1, 1)


def set_result(members) -> dict:
    """The result block for a set answer: sorted unique members, and their count."""
    members = sorted({str(member) for member in members})
    return {"answer": members, "count": len(members)}


# Per building and year: its valid electricity hours, how many of those hours also have a valid air
# temperature at its site, and the site's valid air temperature hours in the whole year.
COVERAGE_QUERY = """
WITH elec AS (
    SELECT DISTINCT building_id, site_id, timestamp_local AS ts
    FROM electricity
    WHERE value IS NOT NULL AND timestamp_local >= $start AND timestamp_local < $end
),
temp AS (
    SELECT DISTINCT site_id, timestamp_local AS ts
    FROM weather
    WHERE air_temperature IS NOT NULL AND timestamp_local >= $start AND timestamp_local < $end
),
site_year AS (
    SELECT site_id, CAST(EXTRACT(year FROM ts) AS INTEGER) AS year, COUNT(*) AS site_temp_hours
    FROM temp GROUP BY 1, 2
),
building_year AS (
    SELECT e.building_id, e.site_id, CAST(EXTRACT(year FROM e.ts) AS INTEGER) AS year,
           COUNT(*) AS elec_hours, COUNT(t.ts) AS elec_hours_with_temp
    FROM elec e LEFT JOIN temp t ON t.site_id = e.site_id AND t.ts = e.ts
    GROUP BY 1, 2, 3
)
SELECT b.building_id, b.building_type_raw, b.floor_area_m2, y.year,
       y.elec_hours, y.elec_hours_with_temp, COALESCE(s.site_temp_hours, 0) AS site_temp_hours
FROM buildings b
JOIN building_year y ON y.building_id = b.building_id
LEFT JOIN site_year s ON s.site_id = b.site_id AND s.year = y.year
WHERE {filters}
ORDER BY b.building_id, y.year
"""


def yearly_coverage(context, years, building_type: str | None = None):
    """Per building and year, electricity and air temperature coverage counts.

    Only buildings with valid floor-area data, and of `building_type` when given.
    """
    filters = [FLOOR_AREA_FILTER]
    params = {"start": datetime(min(years), 1, 1), "end": datetime(max(years) + 1, 1, 1)}
    if building_type is not None:
        filters.append("b.building_type_raw = $building_type")
        params["building_type"] = building_type
    frame = context.sql(COVERAGE_QUERY.format(filters=" AND ".join(filters)), params)
    frame = frame[frame["year"].isin(years)].copy()
    for column in ("year", "elec_hours", "elec_hours_with_temp", "site_temp_hours"):
        frame[column] = frame[column].astype(int)
    frame["year_hours"] = frame["year"].map(year_hours)
    return frame
