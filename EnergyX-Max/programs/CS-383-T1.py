"""EnergyX-Max program for template CS-383-T1.

Using K-Means with {k} clusters, identify the major load-profile groups for {building_type} buildings.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans

def answer(data, **slots):
    k = int(slots['k'])
    building_type = slots['building_type']
    bldgs = data.buildings(building_type=building_type)
    if not bldgs:
        raise ValueError("No buildings found for the given type.")
    readings = data.readings(bldgs)
    cols = [c for c in readings.columns if not readings[c].isna().all()]
    if not cols:
        raise ValueError("No valid consumption data available.")
    X = readings[cols].fillna(0).values
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    unique_labels = sorted(set(labels))
    return {"answer": [str(l) for l in unique_labels], "unit": None}
