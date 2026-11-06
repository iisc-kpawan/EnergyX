"""EnergyX-Max program for template EDA-Q116-T2.

Group all buildings into {k} clusters based solely on their hourly load-profile shapes and report the average floor area for each resulting cluster.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def answer(data, **slots) -> dict:
    k = slots.get('k')
    
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer greater than or equal to 2.")
    
    # Get all building IDs from metadata (metered buildings only)
    try:
        meta_df = data.metadata()
        building_ids = meta_df.index.tolist()
    except Exception:
        raise ValueError("Failed to retrieve building metadata.")

    if not building_ids:
        raise ValueError("No metered buildings found in the dataset.")

    # Retrieve hourly readings for all buildings
    try:
        readings_df = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to retrieve readings: {str(e)}")

    if readings_df.empty:
        raise ValueError("No valid hourly readings found.")

    # Filter out buildings with insufficient data (less than 90% of hours valid in the year range)
    total_hours = len(readings_df)
    valid_counts = readings_df.count()
    valid_ratio = valid_counts / total_hours
    
    # Filter buildings with at least 90% valid data
    qualifying_buildings = valid_ratio[valid_ratio >= 0.9].index.tolist()
    
    if not qualifying_buildings:
        raise ValueError("No buildings meet the 90% data validity threshold.")

    # Prepare the dataset for clustering: normalized hourly load profiles
    # Select only the qualifying buildings
    cluster_data = readings_df.loc[:, qualifying_buildings]
    
    # Normalize each building's profile to 0-1 range (min-max scaling)
    min_vals = cluster_data.min(axis=0)
    max_vals = cluster_data.max(axis=0)
    
    # Avoid division by zero if a building has constant load (unlikely but possible)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 
    
    normalized_profiles = (cluster_data - min_vals) / range_vals
    
    # Flatten the data for K-Means: rows are hours, columns are buildings
    X = normalized_profiles.values.T 

    # Handle NaN values by imputing with median before clustering
    # Since we normalized to 0-1, median is a safe choice (0.5) or we can use 'most_frequent' if applicable
    # However, for continuous data, median is better. Let's use SimpleImputer with strategy='median'.
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Perform K-Means clustering
    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X_imputed)
    except Exception as e:
        raise ValueError(f"Clustering failed: {str(e)}")

    # Map cluster labels to building IDs
    building_to_cluster = {}
    for i, bid in enumerate(qualifying_buildings):
        building_to_cluster[bid] = int(labels[i])

    # Retrieve floor area for each building from metadata
    try:
        meta_df = data.metadata()
        floor_areas = meta_df.loc[building_ids, 'floor_area_m2'].to_dict()
    except Exception as e:
        raise ValueError(f"Failed to retrieve floor areas: {str(e)}")

    # Group buildings by cluster and calculate average floor area
    cluster_stats = {}
    
    for cluster_id in range(k):
        cluster_buildings = [bid for bid, cid in building_to_cluster.items() if cid == cluster_id]
        if not cluster_buildings:
            continue
            
        areas = [floor_areas[bid] for bid in cluster_buildings]
        avg_area = sum(areas) / len(areas)
        
        # Round to 6 decimals as per convention
        avg_area = round(avg_area, 6)
        
        cluster_stats[str(cluster_id)] = avg_area

    return {"answer": cluster_stats, "unit": "m2"}
