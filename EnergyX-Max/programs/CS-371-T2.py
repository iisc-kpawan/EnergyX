"""EnergyX-Max program for template CS-371-T2.

Using K-Means with {k} clusters, identify the cluster containing {building_id} based on normalized hourly consumption.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans

def answer(data, **slots):
    k = int(slots['k'])
    building_id = slots['building_id']
    readings = data.readings([building_id])
    if building_id not in readings.columns:
        raise ValueError("Building not found")
    col = readings[building_id]
    if col.isna().all():
        raise ValueError("No valid readings for building")
    valid = ~col.isna()
    if valid.sum() < 0.9 * len(col):
        raise ValueError("Insufficient valid hours")
    vals = col[valid].values
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    X = norm.reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = km.labels_
    return {"answer": str(km.cluster_centers_[labels[0]]), "unit": None}
