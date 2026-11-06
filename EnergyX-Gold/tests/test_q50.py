"""Q50: DTW TimeSeriesKMeans (k 3) clusters of London education profiles, from BDG2's CSV."""

from __future__ import annotations

import numpy as np
from tslearn.clustering import TimeSeriesKMeans

from test_q48 import london_profiles

TASK = "Q50"


def test_matches_raw_bdg2_file(gold, raw):
    ids, matrix = london_profiles(raw, "Education")
    model = TimeSeriesKMeans(n_clusters=3, metric="dtw", n_init=10, max_iter=50, random_state=0)
    labels = model.fit_predict(matrix[:, :, None])
    groups = [sorted(np.array(ids)[labels == label].tolist()) for label in range(3)]
    groups.sort(key=lambda g: (-len(g), g[0]))

    output = gold(TASK)
    assert output["result"]["answer"] == {"clusters": groups}
    assert output["evidence"]["buildings"] == len(ids)
