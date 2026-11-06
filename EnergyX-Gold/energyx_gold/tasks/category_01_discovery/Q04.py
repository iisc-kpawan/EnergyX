"""Q04: Which office buildings are eligible for weather-normalized analysis in 2016?

Pinned definition: office buildings with valid floor-area data for which at least 95% of 2016's
8,784 hours have a valid electricity reading and at least 95% of those hours have a valid air
temperature at the building's site.

"Those hours" is read as the building's valid electricity hours: the temperature bar is at least
95% of them. An office building has BDG2 primaryspaceusage "Office". The evidence also gives the
set under the other reading, where the site needs a valid air temperature in at least 95% of the
year's hours.

    python -m energyx_gold.tasks.category_01_discovery.Q04
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_01_discovery._helpers import set_result, year_hours, yearly_coverage

TASK_ID = "Q04"
BUILDING_TYPE = "Office"
YEAR = 2016
MIN_COVERAGE_PCT = 95  # compared in integers: 100 * hours >= 95 * base


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    rows = yearly_coverage(context, [YEAR], building_type=BUILDING_TYPE)
    enough_elec = 100 * rows["elec_hours"] >= MIN_COVERAGE_PCT * rows["year_hours"]
    enough_joint = 100 * rows["elec_hours_with_temp"] >= MIN_COVERAGE_PCT * rows["elec_hours"]
    enough_site = 100 * rows["site_temp_hours"] >= MIN_COVERAGE_PCT * rows["year_hours"]
    passing = rows[enough_elec & enough_joint]
    alternative = rows[enough_elec & enough_site]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": set_result(passing["building_id"]),
        "evidence": {
            "definition": {
                "building_type": BUILDING_TYPE,
                "type_column": "building_type_raw",
                "year": YEAR,
                "year_hours": year_hours(YEAR),
                "min_coverage_pct": MIN_COVERAGE_PCT,
                "floor_area": "floor_area_m2 present and > 0",
                "temperature_base": "the building's valid electricity hours in the year",
                "weather_variable": "air_temperature",
            },
            "offices_with_floor_area_and_a_2016_reading": len(rows),
            "offices_passing_electricity_bar": int(enough_elec.sum()),
            "min_elec_hours_among_members": int(passing["elec_hours"].min()) if len(passing) else None,
            "min_joint_coverage_among_members": (
                round(float((passing["elec_hours_with_temp"] / passing["elec_hours"]).min()), 6)
                if len(passing) else None),
            "alternative_site_temperature_over_year_hours": {
                "count": len(alternative),
                "added": sorted(set(alternative["building_id"]) - set(passing["building_id"])),
                "dropped": sorted(set(passing["building_id"]) - set(alternative["building_id"])),
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
