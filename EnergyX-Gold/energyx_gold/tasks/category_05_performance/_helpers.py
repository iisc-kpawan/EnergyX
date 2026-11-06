"""Helpers shared by the category 5 tasks: annual 2017 energy use intensity (EUI).

EUI is a building's valid 2017 kWh divided by its floor area in m2 (BDG2 sqm). A reading is valid
when its value isn't null, so an hour with no row and a ZERO_MASKED row both count as missing.
A building qualifies for a yearly measure when at least 90% of 2017's 8,760 hours are valid.
Valid floor-area data means floor_area_m2 is present and greater than 0.

The primary use label is BDG2's primaryspaceusage, held as building_type_raw. Location comes from
the site: a building is in the United States when its site's country_canonical is 'US'.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from energyx_gold.core.context import GoldContext

YEAR = 2017
YEAR_START = datetime(YEAR, 1, 1)
YEAR_END = datetime(YEAR + 1, 1, 1)  # exclusive
HOURS_IN_YEAR = 8760
MIN_VALID_SHARE = 0.90
USE_LABEL_COLUMN = "building_type_raw"  # BDG2 primaryspaceusage
OFFICE = "Office"
EDUCATION = "Education"
EUI_UNIT = "kWh/m2 per year"
DECIMALS = 6

ANNUAL_QUERY = """
SELECT b.building_id, b.building_type_raw AS use_label, b.floor_area_m2, b.year_built,
       s.country_canonical AS site_country,
       SUM(e.value) AS total_kwh, COUNT(e.value) AS valid_hours
FROM buildings b
JOIN sites s USING (site_id)
JOIN electricity e ON e.building_id = b.building_id
WHERE e.timestamp_local >= $start AND e.timestamp_local < $end
  AND e.value IS NOT NULL{condition}
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1
"""


def rounded(value: float) -> float:
    return round(float(value), DECIMALS)


def annual_eui(context: GoldContext, building_id: str | None = None) -> pd.DataFrame:
    """2017 total kWh, valid hours, coverage and EUI per building, indexed by building_id.

    Buildings with no valid 2017 reading are not in the frame. EUI is NaN where the floor area
    isn't valid. One building only, when asked.
    """
    condition = " AND b.building_id = $building_id" if building_id else ""
    params: dict = {"start": YEAR_START, "end": YEAR_END}
    if building_id:
        params["building_id"] = building_id
    frame = context.sql(ANNUAL_QUERY.format(condition=condition), params).set_index("building_id")
    frame["coverage"] = frame["valid_hours"] / HOURS_IN_YEAR
    area = frame["floor_area_m2"].where(frame["floor_area_m2"] > 0)
    frame["eui"] = frame["total_kwh"] / area
    return frame


def valid_area(frame: pd.DataFrame) -> pd.DataFrame:
    """The buildings with a floor area present and above 0."""
    return frame[frame["floor_area_m2"].notna() & (frame["floor_area_m2"] > 0)]


def full_year(frame: pd.DataFrame) -> pd.DataFrame:
    """The buildings with at least 90% of 2017's hours valid."""
    return frame[frame["valid_hours"] >= MIN_VALID_SHARE * HOURS_IN_YEAR]


def eui_cohort(context: GoldContext) -> pd.DataFrame:
    """Every building with valid floor-area data and at least 90% of 2017's hours valid."""
    return full_year(valid_area(annual_eui(context)))


def of_use(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    return frame[frame["use_label"] == label]


def by_eui(frame: pd.DataFrame) -> pd.DataFrame:
    """Descending EUI; a tie is broken by building_id ascending."""
    return frame.sort_index().sort_values("eui", ascending=False, kind="mergesort")


def ranking(frame: pd.DataFrame, limit: int | None = None) -> list[dict]:
    ordered = by_eui(frame)
    if limit is not None:
        ordered = ordered.head(limit)
    return [
        {
            "rank": position,
            "building_id": str(building_id),
            "eui": rounded(row["eui"]),
            "total_kwh": rounded(row["total_kwh"]),
            "floor_area_m2": rounded(row["floor_area_m2"]),
            "valid_hours": int(row["valid_hours"]),
        }
        for position, (building_id, row) in enumerate(ordered.iterrows(), start=1)
    ]


def percentile(values: pd.Series, share: float) -> float:
    """numpy's default linear interpolation between neighbouring values."""
    return float(values.quantile(share, interpolation="linear"))


def group_numbers(frame: pd.DataFrame) -> dict:
    eui = frame["eui"]
    return {
        "buildings": int(len(frame)),
        "median_eui": rounded(eui.median()),
        "mean_eui": rounded(eui.mean()),
        "p25_eui": rounded(percentile(eui, 0.25)),
        "p75_eui": rounded(percentile(eui, 0.75)),
        "min_eui": rounded(eui.min()),
        "max_eui": rounded(eui.max()),
    }


def period_note() -> str:
    return f"{YEAR_START:%Y-%m-%d %H:%M} to {YEAR_END:%Y-%m-%d %H:%M} (end exclusive), site-local"


def coverage_note() -> str:
    return f"valid hours >= {MIN_VALID_SHARE} x {HOURS_IN_YEAR}"


EUI_NOTE = "sum of valid 2017 readings / floor_area_m2"
AREA_NOTE = "floor_area_m2 present and > 0"


def set_result(members: list[str]) -> dict:
    members = sorted({str(member) for member in members})
    return {"answer": members, "count": len(members)}
