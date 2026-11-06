"""EnergyX-Max program for template CS-003-T2.

What is the total floor area of all {building_type} buildings that have at least one year of complete data?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_type = slots.get('building_type')
    
    # Get all buildings of the specified type
    try:
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found with type '{building_type}'")

    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}'")

    # Determine the global time range by checking readings of the first building
    try:
        sample_readings = data.readings([building_ids[0]])
        global_start = sample_readings.index.min()
        global_end = sample_readings.index.max()
    except Exception:
        raise ValueError("Unable to determine dataset time range.")

    # Define the range of years to check based on the global start and end
    min_year = global_start.year
    max_year = global_end.year
    
    total_floor_area = 0.0
    
    for bldg_id in building_ids:
        try:
            readings_df = data.readings([bldg_id])
        except Exception:
            continue
            
        if readings_df.empty:
            continue
            
        # Get the min and max date from this building's readings to determine its available years
        start_date_bldg = readings_df.index.min()
        end_date_bldg = readings_df.index.max()
        
        # Determine the range of years for this building
        year_start = start_date_bldg.year
        year_end = end_date_bldg.year
        
        # Check each year in the intersection of the building's available years and the global range
        has_complete_year = False
        
        for y in range(year_start, year_end + 1):
            start_ts = pd.Timestamp(f"{y}-01-01")
            end_ts = pd.Timestamp(f"{y+1}-01-01")
            
            # Ensure we are within the global dataset range
            actual_start = max(start_ts, global_start)
            actual_end = min(end_ts, global_end)
            
            if actual_start >= actual_end:
                continue
                
            # Check for missing values in this year's data for this building
            year_data = readings_df.loc[actual_start:actual_end]
            
            # A reading is valid when it is not NaN. 
            # Complete data means no missing hourly readings for any hour in the building's available years (for that specific year).
            if year_data.isna().sum().sum() == 0:
                has_complete_year = True
                break
        
        if has_complete_year:
            # Get floor area from metadata
            try:
                meta = data.metadata()
                floor_area = meta.loc[bldg_id, 'floor_area_m2']
                total_floor_area += floor_area
            except Exception:
                continue
                
    return {"answer": round(total_floor_area, 6), "unit": "m2"}
