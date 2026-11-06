"""Q35: Which two cities in different countries had the most similar average weekday load profiles for office buildings during 2017?

Pinned definition: a city's office profile is the mean of the normalized 2017 weekday profiles of
its office buildings with at least 90% of 2017's hours valid, as in Q34. Only cities with at least
3 such buildings count. For every pair of cities in different countries, the Pearson correlation
between their two profiles. The answer is the pair with the highest correlation; every pair's
correlation is in the evidence.

- The city is sites.city, so the three London sites are one city and Crow and Moose are one Ottawa.
- The correlation is over the 24 hours of the two city profiles.
- The pair is written "CityA|CityB" in alphabetical order. A tie would be broken by that name.

    python -m energyx_gold.tasks.category_07_spatial.Q35
"""

from __future__ import annotations

import itertools

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_07_spatial._helpers import (
    HOURS,
    MIN_COVERAGE,
    OFFICE,
    group_profile,
    qualifying_office_profiles,
    rounded,
)

TASK_ID = "Q35"
YEAR = 2017
MIN_OFFICES_PER_CITY = 3


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    scaled, info = qualifying_office_profiles(context, YEAR)
    named = scaled[scaled["city"].notna()]
    counts = named.groupby("city").size()
    kept = sorted(counts[counts >= MIN_OFFICES_PER_CITY].index)
    left_out = sorted(counts[counts < MIN_OFFICES_PER_CITY].index)

    cities = named[named["city"].isin(kept)]
    profiles = group_profile(cities, "city")
    country = cities.groupby("city")["country"].first()

    pairs = {}
    for left, right in itertools.combinations(kept, 2):
        if country[left] == country[right]:
            continue
        pairs[f"{left}|{right}"] = float(profiles.loc[left].corr(profiles.loc[right]))
    ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    winner = ranked[0][0]

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": winner},
        "evidence": {
            "definition": {
                "period": f"{YEAR}-01-01 to {YEAR}-12-31, site-local",
                "days": "Monday to Friday",
                "building_type": f"BDG2 primaryspaceusage {OFFICE}",
                "min_coverage": MIN_COVERAGE,
                "city": "sites.city",
                "city_profile": "the mean of its offices' profiles, each scaled to 0-1",
                "min_offices_per_city": MIN_OFFICES_PER_CITY,
                "pairs": "cities in different countries",
                "similarity": "Pearson correlation over the 24 hours",
                "tie_break": "pair name",
            },
            "winner_correlation": rounded(ranked[0][1]),
            "runner_up": ranked[1][0] if len(ranked) > 1 else None,
            "runner_up_correlation": rounded(ranked[1][1]) if len(ranked) > 1 else None,
            "cities_scored": kept,
            "cities_left_out": {str(c): int(counts[c]) for c in left_out},
            "offices_by_city": {c: int(counts[c]) for c in kept},
            "country_by_city": {c: str(country[c]) for c in kept},
            "sites_by_city": {c: sorted(set(cities.loc[cities["city"] == c, "site_id"])) for c in kept},
            "left_out_missing_an_hour": info["left_out_missing_an_hour"],
            "correlation_by_pair": {name: rounded(value) for name, value in ranked},
            "scaled_profile_by_city": {c: [rounded(v) for v in profiles.loc[c][HOURS]] for c in kept},
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
