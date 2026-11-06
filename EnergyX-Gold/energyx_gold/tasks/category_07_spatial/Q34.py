"""Q34: How did the average weekday load profiles of office buildings vary across countries during 2017?

Pinned definition: office buildings at sites with a country and with at least 90% of 2017's hours
valid. Each building's weekday mean hourly profile for 2017 is normalized to 0-1, and a country's
profile is the mean of its buildings' normalized profiles. The night-to-day ratio is the mean of
hours 20 to 23 and 0 to 6 over the mean of hours 7 to 19. The country profiles are in the evidence.

- A building's profile is the mean of its valid readings in each hour of the day, Monday to
  Friday of 2017, site-local. A building missing a whole hour of the day would be left out.
- Normalized means scaled by the profile's own minimum and maximum; a flat profile becomes zeros.
- The country is sites.country_canonical. The peak hour is the hour with the highest value in the
  country's mean profile.

    python -m energyx_gold.tasks.category_07_spatial.Q34
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_07_spatial._helpers import (
    DAY_HOURS,
    HOURS,
    MIN_COVERAGE,
    NIGHT_HOURS,
    OFFICE,
    group_profile,
    night_to_day,
    qualifying_office_profiles,
    rounded,
)

TASK_ID = "Q34"
YEAR = 2017


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    scaled, info = qualifying_office_profiles(context, YEAR)
    named = scaled[scaled["country"].notna()]
    profiles = group_profile(named, "country")
    counts = named.groupby("country").size()

    answer, units = {}, {}
    for country in sorted(profiles.index):
        profile = profiles.loc[country]
        answer[f"{country}_peak_hour"] = int(profile.idxmax())
        answer[f"{country}_night_to_day_ratio"] = rounded(night_to_day(profile))
        answer[f"{country}_office_buildings"] = int(counts[country])
        units[f"{country}_peak_hour"] = "hour of day, 0 to 23"
        units[f"{country}_night_to_day_ratio"] = "ratio"
        units[f"{country}_office_buildings"] = "count (buildings)"

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": units},
        "evidence": {
            "definition": {
                "period": f"{YEAR}-01-01 to {YEAR}-12-31, site-local",
                "days": "Monday to Friday",
                "building_type": f"BDG2 primaryspaceusage {OFFICE}",
                "min_coverage": MIN_COVERAGE,
                "profile": "mean valid reading per hour of the day, per building",
                "scaling": "each building's profile scaled to 0-1 by its own minimum and maximum",
                "country_profile": "the mean of its buildings' scaled profiles",
                "night_hours": sorted(NIGHT_HOURS),
                "day_hours": list(DAY_HOURS),
            },
            "qualifying_offices": info["qualifying_offices"],
            "qualifying_offices_at_anonymous_sites": int(len(scaled) - len(named)),
            "left_out_missing_an_hour": info["left_out_missing_an_hour"],
            "flat_profiles": info["flat_profiles"],
            "buildings_by_country": {str(c): int(counts[c]) for c in sorted(profiles.index)},
            "peak_value_by_country": {str(c): rounded(profiles.loc[c].max()) for c in sorted(profiles.index)},
            "scaled_profile_by_country": {
                str(c): [rounded(v) for v in profiles.loc[c][HOURS]] for c in sorted(profiles.index)
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
