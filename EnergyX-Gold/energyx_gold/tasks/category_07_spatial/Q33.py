"""Q33: Which city had the highest median annual electricity EUI for office buildings in 2017?

Pinned definition: annual 2017 EUI for office buildings with valid floor-area data and at least 90%
of 2017's hours valid, grouped by their site's city. Offices at anonymous sites have no city and are
left out, and so are cities with fewer than 5 such offices. The answer is the city with the highest
median; every city's median and size is in the evidence.

- EUI is the year's valid kWh divided by floor_area_m2 (present and greater than 0).
- The city is sites.city, so the three London sites are one city and Crow and Moose are both Ottawa.
- A tie would be broken by city name.

    python -m energyx_gold.tasks.category_07_spatial.Q33
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_07_spatial._helpers import (
    MIN_COVERAGE,
    OFFICE,
    annual_by_building,
    hours_in_year,
    of_type,
    qualifying,
    rounded,
)

TASK_ID = "Q33"
YEAR = 2017
MIN_OFFICES_PER_CITY = 5


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    annual = annual_by_building(context, YEAR)
    offices = of_type(qualifying(annual), OFFICE)
    with_area = offices[offices["eui"].notna()]
    with_city = with_area[with_area["city"].notna()]
    no_city = with_area[with_area["city"].isna()]

    counts = with_city.groupby("city").size()
    medians = with_city.groupby("city")["eui"].median()
    kept = counts[counts >= MIN_OFFICES_PER_CITY].index
    left_out = sorted(counts[counts < MIN_OFFICES_PER_CITY].index)
    ranked = medians[kept].sort_index().sort_values(ascending=False, kind="stable")
    winner = str(ranked.index[0])

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": winner},
        "evidence": {
            "definition": {
                "year": YEAR,
                "building_type": f"BDG2 primaryspaceusage {OFFICE}",
                "eui": "valid 2017 kWh / floor_area_m2, floor_area_m2 > 0",
                "min_coverage": MIN_COVERAGE,
                "hours_in_year": hours_in_year(YEAR),
                "grouping": "sites.city",
                "min_offices_per_city": MIN_OFFICES_PER_CITY,
                "tie_break": "city name",
            },
            "winner_median_eui_kwh_per_m2": rounded(ranked.iloc[0]),
            "runner_up": str(ranked.index[1]) if len(ranked) > 1 else None,
            "runner_up_median_eui_kwh_per_m2": rounded(ranked.iloc[1]) if len(ranked) > 1 else None,
            "qualifying_offices": int(len(offices)),
            "qualifying_offices_without_floor_area": int(len(offices) - len(with_area)),
            "offices_at_anonymous_sites": int(len(no_city)),
            "cities_left_out": {str(city): int(counts[city]) for city in left_out},
            "median_eui_by_city": {
                str(city): {"offices": int(counts[city]), "median_eui_kwh_per_m2": rounded(medians[city])}
                for city in ranked.index
            },
            "sites_by_city": {
                str(city): sorted(set(with_city.loc[with_city["city"] == city, "site_id"])) for city in ranked.index
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
