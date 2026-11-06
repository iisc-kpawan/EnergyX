"""EnergyX-Max program for template EDA-Q049-T1.

Using K-Means with {k} clusters on weekly load profiles for all {building_type} buildings in {country}, identify the top {top_n} clusters representing unobserved shifts in occupancy behavior.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import median_abs_deviation

def answer(data, **slots) -> dict:
    k = int(slots['k'])
    building_type = slots['building_type']
    country = slots['country']
    top_n = int(slots['top_n'])

    # Filter buildings by type and country
    try:
        building_ids = data.buildings(building_type=building_type, country=country)
    except Exception:
        all_bldgs = data.buildings()
        filtered_bldgs = []
        for bid in all_bldgs:
            meta = data.metadata()[bid]
            if meta['building_type'] == building_type and meta['country'] == country:
                filtered_bldgs.append(bid)
        building_ids = filtered_bldgs

    if not building_ids:
        raise ValueError(f"No {building_type} buildings found in {country}.")

    # Get readings for all selected buildings
    try:
        df_readings = data.readings(building_ids)
    except Exception:
        raise ValueError("Failed to retrieve energy readings.")

    if df_readings.empty:
        raise ValueError("No valid readings available for the specified buildings.")

    # Define weekly profile shape (7 days * 24 hours = 168 hours)
    n_hours_per_week = 168
    
    # Initialize arrays to store profiles and their validity counts
    profiles = np.zeros((len(building_ids), n_hours_per_week))
    valid_counts = np.zeros(len(building_ids))

    # Iterate over each building to extract weekly profiles
    for idx, bid in enumerate(building_ids):
        col_data = df_readings[bid]
        
        # Check if the building has enough data to form a reasonable profile
        valid_mask = ~col_data.isna()
        if not valid_mask.any():
            continue
            
        timestamps = df_readings.index
        day_of_week = timestamps.dayofweek
        
        # Initialize weekly profile for this building
        weekly_profile = np.zeros(n_hours_per_week)
        
        for t, val in zip(timestamps[valid_mask], col_data[valid_mask]):
            dow = int(t.dayofweek)
            hour_of_day = int(t.hour)
            weekly_profile[hour_of_day] += val
            
        # Normalize by count to get average profile for this building
        valid_hours_total = np.sum(weekly_profile > 0) # Count of hours with data
        if valid_hours_total > 0:
            profiles[idx, :] = weekly_profile / valid_hours_total
            valid_counts[idx] = valid_hours_total

    # Filter buildings that have some data (valid_counts > 0)
    active_indices = np.where(valid_counts > 0)[0]
    if len(active_indices) == 0:
        raise ValueError("No buildings with sufficient reading history to construct weekly profiles.")
    
    active_profiles = profiles[active_indices, :]
    active_bldg_ids = [building_ids[i] for i in active_indices]

    # Normalize profiles to 0-1 range (min-max scaling) as per convention
    min_vals = active_profiles.min(axis=1, keepdims=True)
    max_vals = active_profiles.max(axis=1, keepdims=True)
    
    # Avoid division by zero if a profile is constant
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0 
    
    normalized_profiles = (active_profiles - min_vals) / ranges

    # Calculate the median profile across all buildings to measure deviation
    median_profile = np.median(normalized_profiles, axis=0)

    # Run K-Means on the normalized profiles
    if len(active_profiles) < k:
        raise ValueError(f"Not enough buildings ({len(active_profiles)}) to form {k} clusters.")

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(normalized_profiles)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # Calculate deviation of each center from the median profile
    center_deviations = np.sqrt(np.sum((centers - median_profile) ** 2, axis=1))
    
    # Sort clusters by deviation descending
    sorted_indices = np.argsort(center_deviations)[::-1]
    
    # Get top_n cluster indices
    top_cluster_indices = sorted_indices[:top_n]
    
    # Prepare the record output
    result_record = {}
    
    for i, idx in enumerate(top_cluster_indices):
        cluster_id = int(idx)
        
        # Calculate stats for this specific cluster centroid
        centroid = centers[idx]
        deviation = center_deviations[idx]
        
        # Identify which buildings belong to this cluster
        cluster_members = np.where(labels == idx)[0]
        member_bldg_ids = [active_bldg_ids[j] for j in cluster_members]
        
        result_record[f"cluster_{cluster_id}"] = {
            "deviation_from_median": round(float(deviation), 6),
            "member_count": len(cluster_members),
            "member_buildings": sorted(member_bldg_ids) # Sorted list of strings
        }

    return {"answer": result_record, "unit": None}
