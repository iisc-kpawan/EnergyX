"""Q46: K-Means (k 3) cluster sizes of Eagle_office_Lamont's complete 2017 days, from BDG2's CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

TASK = "Q46"
BUILDING = "Eagle_office_Lamont"


def sizes(matrix: np.ndarray) -> list[int]:
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(matrix)
    return sorted((int(n) for n in np.bincount(labels)), reverse=True)


def test_matches_raw_bdg2_file(gold, raw):
    frame = raw.electricity_cleaned
    readings = frame.loc[(frame.index >= "2017-01-01") & (frame.index < "2018-01-01"), BUILDING]
    table = pd.DataFrame({"day": readings.index.date, "hour": readings.index.hour, "kwh": readings.to_numpy()})
    wide = table.pivot_table(index="day", columns="hour", values="kwh", aggfunc="first", dropna=False)
    wide = wide.reindex(columns=range(24))
    days = wide[wide.notna().all(axis=1)].to_numpy()

    low, high = days.min(axis=1, keepdims=True), days.max(axis=1, keepdims=True)
    span = high - low
    normalized = np.where(span > 0, (days - low) / np.where(span > 0, span, 1.0), 0.0)

    output = gold(TASK)
    assert output["result"]["answer"] == {"cluster_sizes": sizes(days)}
    assert output["evidence"]["complete_days"] == len(days)
    assert output["evidence"]["cluster_sizes_normalized_days"] == sizes(normalized)
