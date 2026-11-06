"""EnergyX-Max program for template CS-365-T1.

Group {top_n} buildings into clusters using K-Means with {k} clusters based on hourly electricity consumption patterns.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def answer(data, **slots):
    top_n = slots['top_n']
    k = slots['k']
    all_buildings = data.buildings()
    if len(all_buildings) < top_n:
        raise ValueError("Not enough buildings to select.")
    selected_ids = all_buildings[:top_n]
    readings = data.readings(selected_ids)
    if readings.empty:
        raise ValueError("No readings available for selected buildings.")
    norm_profiles = readings.div(readings.sum(axis=0), axis=1).fillna(0)
    X_flat = norm_profiles.values
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X_flat)
    result = {}
    for i in range(k):
        cluster_ids = [selected_ids[j] for j in range(top_n) if labels[j] == i]
        result[f"cluster_{i}"] = sorted(cluster_ids)
    return {"answer": result, "unit": None}
