"""Q48: Cluster the daily electricity load profiles of office buildings in London using
Hierarchical Clustering. Which number of clusters between 2 and 8 gives the highest Silhouette Score?

Definition from data/gold50_questions.yaml:
London office buildings with readings. Each building's mean hourly profile over 2016-2017 is
normalized to 0-1. Agglomerative clustering with Ward linkage on Euclidean distance runs for k = 2
to 8, and the k with the highest silhouette score wins, a tie going to the smaller k. Every k's
score is in the evidence.

- London is the Mouse, Robin and Shrew sites; office is BDG2 primaryspaceusage "Office".
- "With readings" means at least one valid reading in 2016-2017.
- The silhouette score is scikit-learn's, Euclidean. Scores equal at 12 decimals count as a tie.
- Ward linkage has no random part, so the seed does not apply.

    python -m energyx_gold.tasks.category_10_clustering.Q48
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_10_clustering._helpers import (
    best_k,
    cluster_sizes,
    cohort_profiles,
    describe_period,
    period_bounds,
    rounded,
    silhouettes,
    ward_labels,
)

TASK_ID = "Q48"
CITY = "London"
BUILDING_TYPE = "Office"
K_RANGE = range(2, 9)
UNIT = "clusters"


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    start, end = period_bounds(context)
    profiles = cohort_profiles(context, start, end, city=CITY, building_type_raw=BUILDING_TYPE)
    matrix = profiles.to_numpy()

    k, scores, labels = best_k(matrix, lambda c: ward_labels(matrix, c), K_RANGE)
    runner_up = max((c for c in scores if c != k), key=lambda c: (scores[c], -c))
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": int(k), "unit": UNIT},
        "evidence": {
            "definition": {
                "cohort": {"city": CITY, "building_type_raw": BUILDING_TYPE, "with_readings": True},
                "period": describe_period(start, end),
                "profile": "mean valid reading per local hour of day (24 values), min-max scaled to 0-1",
                "agglomerative": {"linkage": "ward", "metric": "euclidean"},
                "k_range": [min(K_RANGE), max(K_RANGE)],
                "selection": "highest silhouette score (Euclidean); a tie goes to the smaller k",
            },
            "buildings": len(profiles),
            "building_ids": list(profiles.index),
            "silhouette_by_k": silhouettes(scores),
            "best_silhouette": rounded(scores[k]),
            "runner_up": {"k": int(runner_up), "silhouette": rounded(scores[runner_up])},
            "cluster_sizes": cluster_sizes(labels[k]),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
