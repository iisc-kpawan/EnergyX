"""Q49: Compare K-Means and Hierarchical Clustering for the daily electricity load profiles of
office buildings in London. Which method gives the better clustering result?

Definition from data/gold50_questions.yaml:
The Q48 profiles. K-Means (n_init 10, random_state 0) and Ward agglomerative clustering each run for
k = 2 to 8 and keep the k with the highest silhouette score, a tie going to the smaller k. The
better method is the one whose best silhouette score is higher.

- The Q48 profiles: London office buildings with at least one valid reading, each building's mean
  valid reading per local hour of day over 2016-2017, min-max scaled to 0-1.
- The silhouette score is scikit-learn's, Euclidean, for both methods.

    python -m energyx_gold.tasks.category_10_clustering.Q49
"""

from __future__ import annotations

from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_10_clustering._helpers import (
    N_INIT,
    best_k,
    cluster_sizes,
    cohort_profiles,
    describe_period,
    kmeans_labels,
    period_bounds,
    rounded,
    silhouettes,
    ward_labels,
)

TASK_ID = "Q49"
CITY = "London"
BUILDING_TYPE = "Office"
K_RANGE = range(2, 9)
KMEANS = "K-Means"
HIERARCHICAL = "Hierarchical"


def main(context: GoldContext | None = None) -> dict:
    from sklearn.metrics import adjusted_rand_score

    context = context or GoldContext()
    start, end = period_bounds(context)
    profiles = cohort_profiles(context, start, end, city=CITY, building_type_raw=BUILDING_TYPE)
    matrix = profiles.to_numpy()

    km_k, km_scores, km_labels = best_k(matrix, lambda c: kmeans_labels(matrix, c, context.seed), K_RANGE)
    ag_k, ag_scores, ag_labels = best_k(matrix, lambda c: ward_labels(matrix, c), K_RANGE)
    if round(km_scores[km_k], 12) == round(ag_scores[ag_k], 12):
        raise ValueError(f"{TASK_ID}: the two methods tie on silhouette score")
    better = KMEANS if km_scores[km_k] > ag_scores[ag_k] else HIERARCHICAL

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {
            "answer": {
                "kmeans_k": int(km_k),
                "kmeans_silhouette": rounded(km_scores[km_k]),
                "hierarchical_k": int(ag_k),
                "hierarchical_silhouette": rounded(ag_scores[ag_k]),
                "better_method": better,
            },
            "units": {"kmeans_k": "clusters", "hierarchical_k": "clusters"},
        },
        "evidence": {
            "definition": {
                "cohort": {"city": CITY, "building_type_raw": BUILDING_TYPE, "with_readings": True},
                "period": describe_period(start, end),
                "profile": "mean valid reading per local hour of day (24 values), min-max scaled to 0-1",
                "kmeans": {"n_init": N_INIT, "random_state": context.seed},
                "agglomerative": {"linkage": "ward", "metric": "euclidean"},
                "k_range": [min(K_RANGE), max(K_RANGE)],
                "selection": "highest silhouette score (Euclidean) per method; a tie goes to the smaller k",
            },
            "buildings": len(profiles),
            "kmeans": {
                "silhouette_by_k": silhouettes(km_scores),
                "cluster_sizes": cluster_sizes(km_labels[km_k]),
            },
            "hierarchical": {
                "silhouette_by_k": silhouettes(ag_scores),
                "cluster_sizes": cluster_sizes(ag_labels[ag_k]),
            },
            "silhouette_difference": rounded(km_scores[km_k] - ag_scores[ag_k]),
            "adjusted_rand_index": rounded(adjusted_rand_score(km_labels[km_k], ag_labels[ag_k])),
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
