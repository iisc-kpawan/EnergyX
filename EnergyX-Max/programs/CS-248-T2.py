"""EnergyX-Max program for template CS-248-T2.

Using K-Means with {k} clusters, identify the top {top_n} sites in {country} where {building_type} buildings show the highest average hourly consumption.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def answer(data, **slots):
    country = slots['country']
    building_type = slots['building_type']
    k = int(slots['k'])
    top_n = int(slots['top_n'])
    
    try:
        bldgs = data.buildings(country=country, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for {country} with type {building_type}")
        
    if not bldgs:
        raise ValueError("No qualifying buildings found.")

    try:
        start, end = data.year_range(2016)
        readings = data.readings(bldgs, start=start, end=end)
        start, end = data.year_range(2017)
        readings_17 = data.readings(bldgs, start=start, end=end)
        readings = pd.concat([readings, readings_17], axis=0)
    except Exception:
        raise ValueError("Failed to retrieve consumption data.")

    valid_mask = ~readings.isna().any(axis=1)
    if not valid_mask.any():
        raise ValueError("No valid hours found in 2016-2017.")
    
    readings = readings[valid_mask]
    
    if len(readings) == 0:
        raise ValueError("No valid data remaining after filtering.")

    hourly_avg = readings.mean(axis=0)
    
    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(hourly_avg.values.reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_
        
        site_cluster_map = {}
        for idx, label in enumerate(labels):
            bldg_id = readings.columns[idx]
            if bldg_id not in site_cluster_map:
                site_cluster_map[bldg_id] = []
            site_cluster_map[bldg_id].append(label)
        
        site_clusters = {}
        for bldg_id, cluster_list in site_cluster_map.items():
            unique_clusters = set(cluster_list)
            if len(unique_clusters) > 1:
                site_clusters[bldg_id] = list(unique_clusters)
            else:
                site_clusters[bldg_id] = [cluster_list[0]]
    except Exception:
        raise ValueError("K-Means clustering failed.")

    site_avg_consumption = {}
    for bldg_id in readings.columns:
        cluster_ids = site_clusters.get(bldg_id, [labels[readings.columns.tolist().index(bldg_id)]])
        if len(cluster_ids) == 1:
            avg_val = hourly_avg[bldg_id]
            site_avg_consumption[bldg_id] = (cluster_ids[0], avg_val)
        else:
            cluster_vals = [hourly_avg[bldg_id]] * len(cluster_ids)
            mean_val = np.mean(cluster_vals)
            site_avg_consumption[bldg_id] = (tuple(sorted(cluster_ids)), mean_val)

    sorted_sites = sorted(site_avg_consumption.items(), key=lambda x: x[1][1], reverse=True)
    top_sites = [site for site, _ in sorted_sites[:top_n]]
    
    result = {}
    for i, site in enumerate(top_sites):
        cluster_id, avg_val = site_avg_consumption[site]
        result[f"site_{i+1}"] = {
            "building_id": site,
            "cluster_id": cluster_id if isinstance(cluster_id, int) else str(cluster_id),
            "avg_hourly_consumption_kwh": round(avg_val, 6)
        }
    
    return {"answer": result, "unit": None}
