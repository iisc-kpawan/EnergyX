"""EnergyX-Max program for template EDA-Q084-T1.

Compare the baseload of {building_id} to the average baseload of {building_type} buildings at the same site.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_id, building_type):
    """
    Compare the baseload of a specific building to the average baseload of buildings
    of the same type at the same site.
    
    Baseload is defined as the mean hourly consumption during non-peak hours.
    Non-peak hours are assumed to be 00:00 to 23:59 excluding peak hours (14:00-18:00).
    """
    
    # Step 1: Get metadata for the target building to find its site_id
    try:
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building {building_id} not found in metadata.")
        target_row = meta.loc[building_id]
        if pd.isna(target_row.get('site_id')):
            raise ValueError(f"Building {building_id} has no site_id.")
        site_id = target_row['site_id']
    except Exception as e:
        raise ValueError(f"Could not retrieve metadata for building {building_id}: {str(e)}")

    # Step 2: Identify comparison buildings (same type, same site)
    try:
        comp_building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception as e:
        raise ValueError(f"No buildings of type '{building_type}' found at site {site_id}.")

    if not comp_building_ids:
        raise ValueError(f"No buildings of type '{building_type}' found at site {site_id}.")

    # Step 3: Get readings for all these buildings
    try:
        df_readings = data.readings(comp_building_ids)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for buildings {comp_building_ids}: {str(e)}")

    if df_readings.empty:
        raise ValueError("No readings available for the specified buildings.")

    # Step 4: Define non-peak hours.
    # Assumption: Peak hours are 14:00 to 18:00 (inclusive). 
    # Non-peak = 00:00-13:59 and 18:00-23:59.
    non_peak_mask = ~df_readings.index.hour.isin([14, 15, 16, 17, 18])
    
    # Step 5: Calculate baseload for each building
    baseloads = {}
    for bid in comp_building_ids:
        col_name = str(bid)
        if col_name not in df_readings.columns:
            continue
            
        series = df_readings[col_name]
        valid_data = series[non_peak_mask]
        
        if len(valid_data) == 0:
            baseloads[bid] = np.nan
        else:
            # Mean of valid readings
            mean_val = valid_data.mean()
            baseloads[bid] = mean_val

    # Step 6: Calculate average baseload for the group
    if len(baseloads) == 0:
        raise ValueError("No valid baseload data calculated.")
    
    group_avg_baseload = np.nanmean(list(baseloads.values()))
    target_baseload = baseloads.get(building_id, np.nan)

    # Prepare the record
    result = {
        "building_id": building_id,
        "site_id": site_id,
        "building_type": building_type,
        "target_baseload_kwh_per_hour": round(float(target_baseload), 6) if not np.isnan(target_baseload) else None,
        "group_avg_baseload_kwh_per_hour": round(float(group_avg_baseload), 6) if not np.isnan(group_avg_baseload) else None,
        "difference_kwh_per_hour": round(float(target_baseload - group_avg_baseload), 6) if (not np.isnan(target_baseload) and not np.isnan(group_avg_baseload)) else None
    }

    return {"answer": result, "unit": "kWh/h"}
