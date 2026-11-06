"""EnergyX-Max program for template CS-084-T1.

Which {building_type} buildings have a nightly average consumption exceeding {threshold_pct}% of their daily peak?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_type, threshold_pct):
    """
    Identifies buildings of a specific type where the average consumption between 00:00 and 06:00
    exceeds a given percentage of the maximum hourly consumption for that building.
    
    Parameters:
    - data: The BDG2 data module.
    - building_type: str, the type of building to filter (e.g., 'Services').
    - threshold_pct: float, the percentage threshold (e.g., 20 for 20%).
    
    Returns:
    - dict: {'answer': sorted list of building IDs meeting the criteria, 'unit': None}
    """
    
    # Retrieve building IDs matching the specified building type
    try:
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        # Fallback if the method signature differs slightly or raises an error unexpectedly
        try:
            metadata_df = data.metadata()
            building_ids = [bid for bid in metadata_df.index 
                            if metadata_df.loc[bid, 'building_type'] == building_type]
        except Exception:
            return {"answer": [], "unit": None}
        
    if not building_ids:
        return {"answer": [], "unit": None}

    # Determine the time range to analyze. Since no specific year is provided in slots,
    # we assume the question refers to the most recent full year available in the dataset.
    # We will scan backwards from a reasonable upper bound (e.g., 2023) to find the latest year with data.
    
    latest_year = None
    for y in range(2015, 2024): 
        s, e = data.year_range(y)
        if s and e:
            try:
                # Check if there is any valid data for at least one building in this year
                df_test = data.readings(building_ids=building_ids[:1], start=s, end=e)
                if not df_test.empty and not df_test.isna().all().all():
                    latest_year = y
                    break
            except Exception:
                continue
        
    if latest_year is None:
        return {"answer": [], "unit": None}
        
    s, e = data.year_range(latest_year)
    
    # Read data for all buildings in the list for the determined year
    try:
        df_readings = data.readings(building_ids=building_ids, start=s, end=e)
    except Exception:
        return {"answer": [], "unit": None}

    if df_readings.empty or len(df_readings.columns) == 0:
        return {"answer": [], "unit": None}

    # Filter columns to only include the buildings we are interested in
    cols_of_interest = [col for col in df_readings.columns if col in building_ids]
    if not cols_of_interest:
        return {"answer": [], "unit": None}

    df_filtered = df_readings[cols_of_interest].copy()

    results = []
    
    # Iterate over each building
    for bid in cols_of_interest:
        series = df_filtered[bid]
        
        # Check validity (NaN handling)
        valid_mask = ~series.isna()
        if not valid_mask.any():
            continue
            
        # Calculate Daily Peak (Max hourly consumption)
        daily_peak = series.max()
        
        if pd.isna(daily_peak):
            continue
            
        # Calculate Nightly Average (00:00 to 06:00)
        # We need to extract hours 0, 1, 2, 3, 4, 5 from the index
        night_hours = [0, 1, 2, 3, 4, 5]
        
        nightly_values = []
        for h in night_hours:
            # Get rows where hour is h
            mask_hour = (series.index.hour == h) & valid_mask
            if mask_hour.any():
                nightly_values.append(series[mask_hour].mean())
        
        if not nightly_values:
            continue
            
        nightly_avg = np.mean(nightly_values)
        
        # Check condition: nightly_avg > threshold_pct% of daily_peak
        # threshold_pct is given as a number (e.g., 20), so we convert to fraction (0.20)
        threshold_fraction = threshold_pct / 100.0
        
        if nightly_avg > (daily_peak * threshold_fraction):
            results.append(bid)

    return {"answer": sorted(results), "unit": None}
