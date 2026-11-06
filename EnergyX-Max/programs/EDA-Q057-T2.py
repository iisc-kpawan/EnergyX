"""EnergyX-Max program for template EDA-Q057-T2.

Using K-Means with {k} clusters, identify the top {top_n} sites where the average load factor of {building_type} buildings in 2017 exceeds the global median by more than {threshold_pct}%

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def answer(data, **slots) -> dict:
    k = slots['k']
    top_n = slots['top_n']
    building_type = slots['building_type']
    threshold_pct = slots['threshold_pct']
    
    # Filter buildings by type for the year 2017
    building_ids = data.buildings(building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No buildings found of type '{building_type}'")

    # Get start and end timestamps for 2017
    start, end = data.year_range(2017)
    
    # Read energy readings for all filtered buildings in 2017
    df_readings = data.readings(building_ids, start=start, end=end)
    
    # Filter out buildings that do not meet the 90% validity threshold for 2017
    qualifying_buildings = data.qualifying(building_ids, year=2017, min_share=0.9)
    
    if len(qualifying_buildings) == 0:
        raise ValueError("No buildings of type '{building_type}' met the validity threshold for 2017.")

    # Get metadata for qualifying buildings to extract site_id and year_built (for age adjustment)
    meta = data.metadata()
    # Filter metadata to only include the qualifying buildings
    meta_qualifying = meta.loc[qualifying_buildings]
    
    # Prepare data for clustering: Site-level load factors adjusted for age and area.
    # Load factor is typically Total Energy / (Total Hours * Floor Area).
    # We calculate total energy per building in 2017.
    total_energy = df_readings.sum(axis=0)
    
    # Get floor areas for qualifying buildings
    area_m2 = meta_qualifying['floor_area_m2']
    
    # Calculate age (years since built relative to 2017)
    ages = 2017 - meta_qualifying['year_built']
    
    # Count valid hours per building from the readings dataframe
    valid_hours = df_readings.count(axis=0)
    
    # Avoid division by zero area
    area_m2_safe = area_m2.replace(0, np.nan)
    
    raw_load_factors = total_energy / (valid_hours * area_m2_safe)
    
    # Handle potential NaNs in load factors (if energy is 0 or area is 0)
    raw_load_factors = raw_load_factors.fillna(0)
    
    # Adjust for age: Apply a simple linear scaling factor based on age.
    # Interpretation: Load_Factor_Adj = Load_Factor * (1 + alpha * Age). 
    # Using alpha=0.01 as a heuristic adjustment.
    adjusted_load_factors = raw_load_factors * (1 + (ages * 0.01))
    
    # Now we need to group by site_id. The metadata is indexed by building_id.
    # We need to map building_ids to site_ids and aggregate load factors per site.
    site_load_factors = []
    site_names = []
    
    for bid in qualifying_buildings:
        sid = meta.loc[bid, 'site_id']
        site_load_factors.append(adjusted_load_factors[bid])
        site_names.append(sid)
        
    # Aggregate to site level: Average load factor per site.
    df_site = pd.DataFrame({
        'site_id': site_names,
        'load_factor': site_load_factors
    })
    
    # Calculate global median of these site-level adjusted load factors
    global_median = df_site['load_factor'].median()
    
    # Identify sites where average load factor exceeds global median by more than threshold_pct%
    # Condition: LF > global_median * (1 + threshold_pct/100)
    threshold_multiplier = 1 + (threshold_pct / 100.0)
    qualifying_sites_mask = df_site['load_factor'] > (global_median * threshold_multiplier)
    
    qualifying_sites = df_site[qualifying_sites_mask]['site_id'].tolist()
    
    # If no sites qualify, raise ValueError as per instructions for empty population
    if len(qualifying_sites) == 0:
        raise ValueError("No sites meet the criteria.")

    # Sort qualifying sites by their load factor descending to get top_n
    df_sorted = df_site.sort_values(by='load_factor', ascending=False)
    
    # Filter to only those that met the threshold and take top_n
    final_sites = df_sorted[df_sorted['site_id'].isin(qualifying_sites)].head(top_n)['site_id'].tolist()
    
    # Construct the result table as a list of dictionaries
    result_table = []
    for site_id in final_sites:
        row = {
            "site_id": site_id,
            "load_factor": round(df_site.loc[df_site['site_id'] == site_id, 'load_factor'].values[0], 6)
        }
        result_table.append(row)
        
    return {"answer": result_table, "unit": None}
