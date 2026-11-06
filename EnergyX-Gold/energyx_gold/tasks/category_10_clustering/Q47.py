"""Q47: Which five office buildings had the most similar average daily electricity load profile to
`Eagle_office_Lamont` during 2017?

Definition from data/gold50_questions.yaml:
Office buildings other than Eagle_office_Lamont with at least 90% of 2017's hours valid. Each
building's 2017 mean hourly profile is normalized to 0-1. Similarity is the Euclidean distance
between a building's normalized profile and Eagle_office_Lamont's. The five closest qualify; the
ranking and the Pearson-correlation ranking are in the evidence.

- Office is BDG2 primaryspaceusage "Office", at any site.
- 2017 has 8,760 local hours, so a candidate needs at least 7,884 valid hours.
- Eagle_office_Lamont's own profile uses all its valid 2017 readings; the 90% rule is for candidates.
- Ties in distance go to the smaller building_id.

    python -m energyx_gold.tasks.category_10_clustering.Q47
"""

from __future__ import annotations

import numpy as np

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_10_clustering._helpers import (
    describe_period,
    rounded,
    scaled_profiles,
    year_bounds,
)

TASK_ID = "Q47"
TARGET = "Eagle_office_Lamont"
BUILDING_TYPE = "Office"
YEAR = 2017
COVERAGE = 0.9
TOP = 5
SHOWN = 10  # rows of each ranking kept in the evidence


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    start, end = year_bounds(YEAR)
    offices = series.cohort(context, building_type_raw=BUILDING_TYPE)
    hourly = series.hourly(context, offices, start, end)

    hours = len(hourly.index)
    valid = hourly.notna().sum()
    candidates = sorted(b for b in offices if b != TARGET and valid[b] >= COVERAGE * hours)
    profiles = scaled_profiles(hourly[[TARGET, *candidates]])
    target = profiles.loc[TARGET].to_numpy()
    others = profiles.loc[candidates].to_numpy()

    distance = np.sqrt(((others - target) ** 2).sum(axis=1))
    by_distance = sorted(zip(candidates, distance), key=lambda row: (row[1], row[0]))
    pearson = [(b, float(np.corrcoef(target, row)[0, 1]) if row.std() > 0 else float("nan"))
               for b, row in zip(candidates, others)]
    by_pearson = sorted((r for r in pearson if not np.isnan(r[1])), key=lambda row: (-row[1], row[0]))

    answer = sorted(b for b, _ in by_distance[:TOP])
    pearson_top = [b for b, _ in by_pearson[:TOP]]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "count": len(answer)},
        "evidence": {
            "definition": {
                "target": TARGET,
                "cohort": {"building_type_raw": BUILDING_TYPE, "min_valid_share_2017": COVERAGE},
                "period": describe_period(start, end),
                "profile": "mean valid reading per local hour of day (24 values), min-max scaled to 0-1",
                "similarity": "Euclidean distance between scaled profiles; ties go to the smaller building_id",
            },
            "office_buildings_with_readings": len(offices),
            "candidates": len(candidates),
            "target_valid_hours_2017": int(valid[TARGET]),
            "hours_in_year": hours,
            "euclidean_ranking": [{"building_id": b, "distance": rounded(d)} for b, d in by_distance[:SHOWN]],
            "pearson_ranking": [{"building_id": b, "pearson_r": rounded(r)} for b, r in by_pearson[:SHOWN]],
            "pearson_top5_overlap": sorted(set(answer) & set(pearson_top)),
            "sixth_distance_gap": rounded(by_distance[TOP][1] - by_distance[TOP - 1][1]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
