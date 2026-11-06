"""EnergyX-Max program for template CS-037-T1.

Which buildings have more than {threshold_pct}% missing hourly electricity consumption readings in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    threshold_pct = slots['threshold_pct']
    year = slots['year']
    
    # Get all building IDs from metadata
    all_building_ids = list(data.metadata().index)
    
    if not all_building_ids:
        raise ValueError("No metered buildings found in the dataset.")

    # Determine start and end timestamps for the year
    start_ts, end_ts = data.year_range(year)
    
    # Readings for all buildings for the specified year
    readings_df = data.readings(all_building_ids, start=start_ts, end=end_ts)
    
    # Filter to only include hours within the specific year (just in case year_range isn't perfectly tight or logic needs it)
    # The read function usually handles this, but let's ensure we are looking at the right window.
    # readings_df index is DatetimeIndex.
    
    valid_count = 0
    total_hours_in_year = (end_ts - start_ts).total_seconds() / 3600
    
    missing_threshold = threshold_pct / 100.0
    
    buildings_with_high_missing = []
    
    for building_id in all_building_ids:
        if building_id not in readings_df.columns:
            continue
            
        col_data = readings_df[building_id]
        
        # Count total hours available in the dataframe for this building within the year range
        # Note: data.readings returns a DataFrame with complete hourly DatetimeIndex. 
        # If a building has no data at all, it might not be in columns or have all NaNs.
        # We assume 'readings' returns columns only for buildings that exist, but might be all NaN if no metering.
        
        total_hours = len(col_data)
        if total_hours == 0:
            continue
            
        # Count valid readings (not NaN)
        valid_readings = col_data.notna().sum()
        
        # Calculate missing percentage
        # Missing count = Total - Valid
        missing_count = total_hours - valid_readings
        
        if total_hours > 0:
            missing_pct = (missing_count / total_hours) * 100
            
            # Check if missing percentage is strictly greater than threshold_pct
            if missing_pct > threshold_pct:
                buildings_with_high_missing.append(building_id)
    
    # Sort the list of building IDs
    result_list = sorted(buildings_with_high_missing)
    
    return {"answer": result_list, "unit": None}
