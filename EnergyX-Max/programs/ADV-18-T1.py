"""EnergyX-Max program for template ADV-18-T1.

What is the Silhouette Score for K-Means clusters with {k} clusters on hourly electricity consumption at the {site_id} site when weekday and weekend profiles are analyzed separately?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def answer(data, **slots) -> dict:
    k = slots.get('k')
    site_id = slots.get('site_id')
    
    # Validate inputs
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer greater than or equal to 2.")
    if pd.isna(site_id) or site_id is None:
        raise ValueError("site_id cannot be empty or None.")

    # Get building IDs for the specified site
    try:
        building_ids = data.buildings(site_id=site_id)
    except Exception as e:
        raise ValueError(f"Could not retrieve buildings for site_id '{site_id}': {str(e)}")
    
    if not building_ids:
        raise ValueError(f"No metered buildings found for site_id '{site_id}'.")

    # Get readings for these buildings. We need a wide format where columns are building IDs.
    # The read function returns a DataFrame with DatetimeIndex and one column per building.
    try:
        df_readings = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for buildings at site '{site_id}': {str(e)}")

    if df_readings.empty:
        raise ValueError("No hourly electricity consumption data found for the specified buildings.")

    # Ensure we have a complete time range. The problem implies analyzing the available data.
    # We assume the data covers enough hours to define weekdays and weekends.
    
    # Separate weekday and weekend hours
    # DatetimeIndex has .dayofweek attribute (0=Monday, 6=Sunday)
    # Weekdays: 0-4, Weekends: 5-6
    
    df_weekday = df_readings[df_readings.index.dayofweek < 5].copy()
    df_weekend = df_readings[df_readings.index.dayofweek >= 5].copy()

    # Function to compute silhouette score for a given subset and k
    def compute_silhouette(df_subset, n_clusters):
        if df_subset.empty:
            return None
        
        # Flatten the dataframe values (excluding index) into a list/array
        X = df_subset.values.flatten()
        
        # Check if there are enough data points for clustering
        if len(X) < n_clusters * 2: # Very basic check, though sklearn handles it
             return None

        # Handle NaN values by removing them before clustering
        valid_mask = ~np.isnan(X)
        X_valid = X[valid_mask]
        
        if len(X_valid) < n_clusters * 2:
            return None
            
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            labels = kmeans.fit_predict(X_valid.reshape(-1, 1)) # Reshape to 2D for sklearn
            
            if len(np.unique(labels)) < 2:
                return None
                
            score = silhouette_score(X_valid.reshape(-1, 1), labels)
            return score
        except Exception as e:
            # If clustering fails (e.g., too few points relative to k), return None
            return None

    # Compute scores
    score_weekday = compute_silhouette(df_weekday, k)
    score_weekend = compute_silhouette(df_weekend, k)

    result = {}
    
    if score_weekday is not None:
        result['weekday'] = round(score_weekday, 6)
    else:
        result['weekday'] = None
        
    if score_weekend is not None:
        result['weekend'] = round(score_weekend, 6)
    else:
        result['weekend'] = None

    return {"answer": result, "unit": None}
