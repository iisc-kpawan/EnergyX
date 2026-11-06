"""Q50: Using TimeSeriesKMeans with DTW distance and k=3, cluster the normalized daily electricity
load profiles of education buildings in London. Which buildings belong to each cluster?

Definition from data/gold50_questions.yaml:
London education buildings with readings. Each building's mean hourly profile over 2016-2017 is
normalized to 0-1. TimeSeriesKMeans (tslearn, metric dtw, k 3, n_init 10, max_iter 50,
random_state 0) clusters the profiles. The answer gives each cluster's sorted members, with the
clusters ordered by size and then by first member.

- London is the Mouse, Robin and Shrew sites; education is BDG2 primaryspaceusage "Education".
- "With readings" means at least one valid reading in 2016-2017.
- Clusters are ordered largest first; equal sizes go by their first (smallest) building_id.

    python -m energyx_gold.tasks.category_10_clustering.Q50
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_10_clustering._helpers import (
    MAX_ITER,
    N_INIT,
    cohort_profiles,
    describe_period,
    dtw_kmeans_labels,
    dtw_silhouette,
    members,
    period_bounds,
    rounded,
)

TASK_ID = "Q50"
CITY = "London"
BUILDING_TYPE = "Education"
K = 3


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    start, end = period_bounds(context)
    profiles = cohort_profiles(context, start, end, city=CITY, building_type_raw=BUILDING_TYPE)
    ids = list(profiles.index)
    matrix = profiles.to_numpy()[:, :, None]

    labels = dtw_kmeans_labels(matrix, K, context.seed)
    clusters = members(ids, labels)
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": {"clusters": clusters}},
        "evidence": {
            "definition": {
                "cohort": {"city": CITY, "building_type_raw": BUILDING_TYPE, "with_readings": True},
                "period": describe_period(start, end),
                "profile": "mean valid reading per local hour of day (24 values), min-max scaled to 0-1",
                "time_series_kmeans": {
                    "n_clusters": K,
                    "metric": "dtw",
                    "n_init": N_INIT,
                    "max_iter": MAX_ITER,
                    "random_state": context.seed,
                },
                "order": "clusters by size, largest first, then by first member; members sorted",
            },
            "buildings": len(ids),
            "cluster_sizes": [len(c) for c in clusters],
            "dtw_silhouette": rounded(dtw_silhouette(matrix, labels)),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
