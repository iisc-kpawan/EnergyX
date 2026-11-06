"""EnergyX-Max program for template CS-368-T2.

Using K-Means with {k} clusters, identify the cluster containing {building_id}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans

def answer(data, **slots):
    k = int(slots['k'])
    target_building_id = slots['building_id']
    
    all_buildings = data.buildings()
    if not all_buildings:
        raise ValueError("No buildings found in the dataset.")
        
    readings_df = data.readings(all_buildings)
    
    if target_building_id not in readings_df.columns:
        raise ValueError(f"Building {target_building_id} not found.")
        
    cluster_labels = KMeans(n_clusters=k, random_state=0).fit_predict(readings_df.dropna(axis=1))
    
    try:
        label = cluster_labels[readings_df.columns.get_loc(target_building_id)]
    except KeyError:
        raise ValueError(f"Building {target_building_id} not found in readings.")
        
    return {"answer": str(label), "unit": None}
