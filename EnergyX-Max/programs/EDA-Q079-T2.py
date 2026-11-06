"""EnergyX-Max program for template EDA-Q079-T2.

Using K-Means with {k} clusters, identify the top {top_n} {building_type} buildings that deviate most from the average daily load shape of {building_type_2} buildings.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def answer(data, **slots) -> dict:
    k = slots.get('k')
    top_n = slots.get('top_n')
    building_type = slots.get('building_type')
    building_type_2 = slots.get('building_type_2')

    # Retrieve buildings for the reference type (building_type_2)
    ref_building_ids = data.buildings(building_type=building_type_2)
    
    if not ref_building_ids:
        raise ValueError(f"No {building_type_2} buildings found.")

    # Retrieve buildings for the target type (building_type)
    target_building_ids = data.buildings(building_type=building_type)
    
    if not target_building_ids:
        raise ValueError(f"No {building_type} buildings found.")

    # Read hourly energy data for reference buildings
    ref_readings = data.readings(ref_building_ids)
    
    # Read hourly energy data for target buildings
    target_readings = data.readings(target_building_ids)

    # Determine the common date range based on available data in both sets
    start_date = ref_readings.index.min()
    end_date = ref_readings.index.max() + pd.Timedelta(hours=1) # exclusive end
    
    # Filter to a common range if necessary, but usually we use the full available range.
    # However, to compute a robust "average daily load shape", we need consistent days.
    # Let's ensure we have at least one day of data.
    if (end_date - start_date).days < 1:
        raise ValueError("Insufficient data duration to compute daily profiles.")

    # Function to compute daily profile for a single building series
    def get_daily_profile(series):
        # Resample to daily mean, handling NaNs by skipping days with no data if necessary?
        # The problem implies "daily load shape". Usually, this means the pattern within a day.
        # We take the mean of each day. If a day has no data, it's skipped or treated as 0?
        # Standard practice for profile analysis: average over available days.
        daily_means = series.resample('D').mean()
        return daily_means

    # Compute daily profiles for reference buildings
    ref_daily_profiles = {}
    valid_ref_count = 0
    
    for bid in ref_building_ids:
        if bid not in target_readings.columns and bid not in ref_readings.columns:
            continue
            
        profile = get_daily_profile(ref_readings[bid])
        if not profile.empty:
            # Normalize to 0-1 range based on min and max of this specific building's profile
            # Or normalize based on the global min/max? 
            # "normalized profile is scaled to 0-1 by its own min and max" -> Convention says "its own".
            if profile.max() > profile.min():
                norm_profile = (profile - profile.min()) / (profile.max() - profile.min())
            else:
                norm_profile = pd.Series([0.5] * len(profile), index=profile.index) # Flat line if constant
            ref_daily_profiles[bid] = norm_profile
            valid_ref_count += 1

    if valid_ref_count == 0:
        raise ValueError("No valid daily profiles could be computed for reference buildings.")

    # Compute the mean profile of the reference type
    # Align all profiles to a common index (the union of days present in any reference building)
    aligned_profiles = [ref_daily_profiles[bid] for bid in ref_building_ids if bid in ref_daily_profiles]
    
    if not aligned_profiles:
        raise ValueError("No valid daily profiles available for reference buildings.")

    # Create a DataFrame from the list of series to compute mean
    profile_df = pd.DataFrame(aligned_profiles)
    # Fill NaNs with 0 before mean? Or skip days? 
    # Convention: "Sums and means use valid readings only."
    # If a day is missing for a building, it shouldn't contribute to the sum for that day.
    # So we compute mean ignoring NaNs per column (day).
    mean_profile = profile_df.mean(axis=0)

    # Compute daily profiles for target buildings
    target_daily_profiles = {}
    valid_target_count = 0
    
    for bid in target_building_ids:
        if bid not in ref_readings.columns and bid not in target_readings.columns:
            continue
            
        profile = get_daily_profile(target_readings[bid])
        if not profile.empty:
            # We need to align the target profile with the mean profile index to compute distance.
            # The mean_profile index contains all days present in any reference building.
            # We should intersect the target profile index with the mean_profile index.
            common_index = mean_profile.index.intersection(profile.index)
            if len(common_index) == 0:
                continue
            
            target_norm_profile = profile.loc[common_index]
            
            # Normalize this specific building's profile to 0-1 by its own min and max
            if target_norm_profile.max() > target_norm_profile.min():
                norm_profile = (target_norm_profile - target_norm_profile.min()) / \
                               (target_norm_profile.max() - target_norm_profile.min())
            else:
                norm_profile = pd.Series([0.5] * len(target_norm_profile), index=target_norm_profile.index)
            
            target_daily_profiles[bid] = norm_profile
            valid_target_count += 1

    if valid_target_count == 0:
        raise ValueError("No valid daily profiles could be computed for target buildings.")

    # Compute Euclidean distance between each target building's profile and the mean profile
    # Ensure indices match
    distances = {}
    for bid, profile in target_daily_profiles.items():
        # Align with mean_profile
        common_idx = mean_profile.index.intersection(profile.index)
        if len(common_idx) == 0:
            continue
        
        p_mean = mean_profile.loc[common_idx]
        p_target = profile.loc[common_idx]
        
        dist = np.sqrt(np.sum((p_mean - p_target) ** 2))
        distances[bid] = dist

    if not distances:
        raise ValueError("Could not compute deviations for any target building.")

    # Sort by distance descending and get top_n
    sorted_buildings = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    top_buildings = [bid for bid, _ in sorted_buildings[:top_n]]

    return {"answer": top_buildings, "unit": None}
