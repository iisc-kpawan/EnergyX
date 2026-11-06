"""EnergyX-Max program for template CS-057-T1.

Which buildings have more than {threshold_pct}% missing hourly electricity consumption readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    threshold_pct = slots.get('threshold_pct')
    
    # Get all building IDs from metadata
    all_building_ids = list(data.metadata().index)
    
    if not all_building_ids:
        raise ValueError("No metered buildings found in the dataset.")

    missing_buildings = []
    
    for bid in all_building_ids:
        # Read hourly consumption data for this building for the entire available range
        # Since no specific date range is requested, we fetch all available data.
        # The read method handles start/end=None by returning the full available history.
        readings_df = data.readings([bid])
        
        if readings_df.empty:
            continue
            
        # Calculate total number of hours in the dataset for this building
        total_hours = len(readings_df)
        
        if total_hours == 0:
            continue
            
        # Count missing (NaN) values
        missing_count = readings_df.isna().sum().iloc[0]
        
        # Calculate percentage of missing readings
        missing_pct = (missing_count / total_hours) * 100
        
        # Check if it exceeds the threshold
        if missing_pct > threshold_pct:
            missing_buildings.append(bid)
    
    # Sort the list of building IDs as per convention for sets/lists in this context
    missing_buildings.sort()
    
    return {"answer": missing_buildings, "unit": None}
