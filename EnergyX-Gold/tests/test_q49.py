"""Q49: K-Means against Ward agglomerative clustering on the London office profiles, k 2 to 8."""

from __future__ import annotations

import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from test_q48 import london_profiles

TASK = "Q49"


def best(matrix, fit) -> tuple[int, float]:
    scores = {k: float(silhouette_score(matrix, fit(k))) for k in range(2, 9)}
    k = max(scores, key=lambda c: (scores[c], -c))
    return k, scores[k]


def test_matches_raw_bdg2_file(gold, raw):
    _, matrix = london_profiles(raw, "Office")
    km_k, km_score = best(matrix, lambda k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix))
    ag_k, ag_score = best(
        matrix, lambda k: AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(matrix)
    )

    answer = gold(TASK)["result"]["answer"]
    assert answer["kmeans_k"] == km_k
    assert answer["hierarchical_k"] == ag_k
    assert answer["kmeans_silhouette"] == pytest.approx(km_score, rel=1e-6, abs=1e-6)
    assert answer["hierarchical_silhouette"] == pytest.approx(ag_score, rel=1e-6, abs=1e-6)
    assert answer["better_method"] == ("K-Means" if km_score > ag_score else "Hierarchical")
