"""Q46: Cluster the daily electricity load profiles of `Eagle_office_Lamont` during 2017 using
K-Means with k=3. How many days belong to each cluster?

Definition from data/gold50_questions.yaml:
Eagle_office_Lamont's complete days of 2017 (24 valid readings each), each as a 24-value vector of
kWh. K-Means (scikit-learn, k 3, n_init 10, random_state 0) on those vectors. The answer lists the
three cluster sizes from largest to smallest. The evidence gives the cluster centres and the sizes
when each day is first normalized to 0-1.

- A day is a local calendar day of 2017, hours 0 to 23. Masked zeros and absent hours are missing.
- The vectors are raw kWh, not scaled. The normalized variant scales each day by its own minimum
  and maximum; a flat day becomes all zeros.

    python -m energyx_gold.tasks.category_10_clustering.Q46
"""

from __future__ import annotations

import numpy as np

from energyx_gold.analytics import clustering, series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_10_clustering._helpers import (
    N_INIT,
    cluster_sizes,
    complete_days,
    describe_period,
    kmeans_model,
    rounded,
    year_bounds,
)

TASK_ID = "Q46"
BUILDING = "Eagle_office_Lamont"
YEAR = 2017
K = 3


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    start, end = year_bounds(YEAR)
    days = complete_days(series.hourly(context, [BUILDING], start, end)[BUILDING])
    model = kmeans_model(days.to_numpy(), K, context.seed)
    sizes = cluster_sizes(model.labels_)

    counts = np.bincount(model.labels_, minlength=K)
    order = sorted(range(K), key=lambda c: (-counts[c], c))
    centres = [
        {"days": int(counts[c]), "centre_kwh": [rounded(v) for v in model.cluster_centers_[c]]}
        for c in order
    ]
    normalized = clustering.minmax_rows(days).to_numpy()
    normalized_sizes = cluster_sizes(kmeans_model(normalized, K, context.seed).labels_)

    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": {"cluster_sizes": sizes}, "units": {"cluster_sizes": "days"}},
        "evidence": {
            "definition": {
                "building_id": BUILDING,
                "period": describe_period(start, end),
                "rows": "each complete local day, 24 valid readings, hour 0 to hour 23, raw kWh",
                "kmeans": {"n_clusters": K, "n_init": N_INIT, "random_state": context.seed},
            },
            "complete_days": int(len(days)),
            "days_in_year": int((end - start).days),
            "clusters_largest_first": centres,
            "cluster_sizes_normalized_days": normalized_sizes,
            "normalized_note": "each day min-max scaled to 0-1 before K-Means, same settings",
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
