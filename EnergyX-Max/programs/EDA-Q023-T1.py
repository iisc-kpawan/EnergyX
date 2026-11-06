"""EnergyX-Max program for template EDA-Q023-T1.

What is the average percentage of missing hourly electricity readings for {building_id} in {year}, and how does this affect the calculated annual Energy Use Intensity (EUI) if imputed versus excluded?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    import numpy as np
    import pandas as pd
    
    # Retrieve metadata for the specific building
    meta = data.metadata()
    
    if building_id not in meta.index:
        raise ValueError(f"Building ID '{building_id}' not found in metadata.")
        
    site_id = meta.loc[building_id, 'site_id']
    floor_area_m2 = meta.loc[building_id, 'floor_area_m2']
    
    # Define the time range for the year
    start_ts, end_ts = data.year_range(year)
    
    # Get readings for the building
    try:
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to retrieve readings for {building_id}: {str(e)}")
        
    if len(readings_df) == 0 or building_id not in readings_df.columns:
        raise ValueError("No data available for the specified building and year.")

    # Calculate missing percentage
    total_hours = (end_ts - start_ts).total_seconds() / 3600
    valid_readings = readings_df[building_id].notna().sum()
    missing_count = total_hours - valid_readings
    missing_pct = (missing_count / total_hours) * 100
    
    # Calculate EUI excluding missing values
    energy_excluded = readings_df[building_id].dropna().sum()
    eui_excluded = None
    if floor_area_m2 > 0 and valid_readings > 0:
        eui_excluded = (energy_excluded / floor_area_m2) * (total_hours / valid_readings)
        # Note: The standard EUI calculation usually sums total energy over the year. 
        # If we exclude missing, we are effectively calculating EUI based on available hours only? 
        # Or do we scale it back to annual? 
        # Standard practice for "excluded": Sum of valid readings / Area. This is an average intensity over valid hours.
        # However, the question asks for "calculated annual EUI". 
        # If we exclude missing, we don't have the full year's energy. 
        # Let's assume the metric requested is: (Sum of Valid Energy) / Floor Area. 
        # This represents the intensity observed during metered hours.
        eui_excluded = energy_excluded / floor_area_m2
        
    # Calculate EUI imputing missing values
    # Simple imputation: fill NaN with the mean of valid readings for that building/year
    mean_valid_reading = readings_df[building_id].dropna().mean()
    if pd.isna(mean_valid_reading):
        mean_valid_reading = 0
        
    readings_imputed = readings_df[building_id].fillna(mean_valid_reading)
    energy_imputed = readings_imputed.sum()
    
    eui_imputed = None
    if floor_area_m2 > 0:
        eui_imputed = energy_imputed / floor_area_m2

    # Prepare the record
    result = {
        "missing_percentage": round(missing_pct, 6),
        "eui_excluded_kwh_per_m2": round(eui_excluded, 6) if eui_excluded is not None else None,
        "eui_imputed_kwh_per_m2": round(eui_imputed, 6) if eui_imputed is not None else None
    }
    
    return {"answer": result, "unit": None}
