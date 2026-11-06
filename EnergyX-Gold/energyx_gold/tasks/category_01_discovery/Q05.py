"""Q05: Which buildings are eligible for weather-normalized annual analysis?

Pinned definition: buildings of any type with valid floor-area data and one calendar year (2016 or
2017) in which at least 80% of the year's hours have a valid electricity reading and at least 80%
of the year's hours have a valid air temperature at the building's site.

Both bars are measured against the same year's hours (8,784 in 2016, 8,760 in 2017), and both
must hold in the same year.

    python -m energyx_gold.tasks.category_01_discovery.Q05
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_01_discovery._helpers import (
    YEARS,
    set_result,
    year_hours,
    yearly_coverage,
)

TASK_ID = "Q05"
MIN_COVERAGE_PCT = 80  # compared in integers: 100 * hours >= 80 * year hours


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = yearly_coverage(context, list(YEARS))
    enough_elec = 100 * rows["elec_hours"] >= MIN_COVERAGE_PCT * rows["year_hours"]
    enough_site = 100 * rows["site_temp_hours"] >= MIN_COVERAGE_PCT * rows["year_hours"]
    enough_joint = 100 * rows["elec_hours_with_temp"] >= MIN_COVERAGE_PCT * rows["year_hours"]
    passing = rows[enough_elec & enough_site]
    members = set(passing["building_id"])
    by_year = {str(y): sorted(set(passing.loc[passing["year"] == y, "building_id"])) for y in YEARS}
    joint = set(rows.loc[enough_elec & enough_joint, "building_id"])
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(members),
        "evidence": {
            "definition": {
                "building_type": "any",
                "years": list(YEARS),
                "hours_per_year": {str(y): year_hours(y) for y in YEARS},
                "min_coverage_pct": MIN_COVERAGE_PCT,
                "floor_area": "floor_area_m2 present and > 0",
                "weather_variable": "air_temperature",
                "same_year_for_both_bars": True,
            },
            "buildings_with_floor_area_and_a_reading": int(rows["building_id"].nunique()),
            "members_qualifying_in_2016": len(by_year["2016"]),
            "members_qualifying_in_2017": len(by_year["2017"]),
            "members_qualifying_in_both_years": len(set(by_year["2016"]) & set(by_year["2017"])),
            "alternative_hours_with_both_over_year_hours": {
                "count": len(joint),
                "dropped": sorted(members - joint),
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
