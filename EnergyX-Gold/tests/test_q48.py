"""Q48: the Ward agglomerative k (2 to 8) with the best silhouette for London office profiles."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

TASK = "Q48"


def london_profiles(raw, space_usage: str) -> tuple[list[str], np.ndarray]:
    meta = raw.metadata
    frame = raw.electricity_cleaned
    wanted = meta.loc[(meta["City"] == "London") & (meta["primaryspaceusage"] == space_usage), "building_id"]
    ids = sorted(b for b in wanted if b in frame.columns and frame[b].notna().any())
    means = frame[ids].groupby(frame.index.hour).mean().T.to_numpy()
    assert not np.isnan(means).any()
    low, high = means.min(axis=1, keepdims=True), means.max(axis=1, keepdims=True)
    span = high - low
    return ids, np.where(span > 0, (means - low) / np.where(span > 0, span, 1.0), 0.0)


def test_matches_raw_bdg2_file(gold, raw):
    ids, matrix = london_profiles(raw, "Office")
    scores = {}
    for k in range(2, 9):
        labels = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward").fit_predict(matrix)
        scores[k] = float(silhouette_score(matrix, labels))
    best = max(scores, key=lambda k: (scores[k], -k))

    output = gold(TASK)
    assert output["result"] == {"answer": best, "unit": "clusters"}
    evidence = output["evidence"]
    assert evidence["buildings"] == len(ids)
    for k, score in scores.items():
        assert evidence["silhouette_by_k"][str(k)] == pytest.approx(score, rel=1e-6, abs=1e-6)
