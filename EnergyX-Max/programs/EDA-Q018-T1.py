"""EnergyX-Max program for template EDA-Q018-T1.

What percentage of electricity consumption timestamps are missing for {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Retrieve readings for the specific building
    # The read method returns a wide DataFrame where columns are building IDs and rows are timestamps.
    # If the building is not found or has no data, we need to handle it.
    try:
        df = data.readings([building_id])
    except Exception:
        raise ValueError(f"No data found for building_id: {building_id}")

    if df.empty or building_id not in df.columns:
        # If the column exists but is empty, or the dataframe is empty
        if building_id not in df.columns:
            raise ValueError(f"Building ID '{building_id}' not found in dataset.")
        
        # Check if there are any valid readings at all for this building
        # If the column exists but has no non-NaN values, it's 100% missing (or just empty)
        # However, usually read() returns a dataframe with the column. 
        # Let's assume if the column is present, we calculate based on that.
        
    # Calculate total number of timestamps in the dataset for this building
    # The DatetimeIndex represents the total count of hours requested/available
    total_timestamps = len(df.index)
    
    if total_timestamps == 0:
        raise ValueError("No timestamps available for the specified building.")

    # Count missing values (NaN) in the column corresponding to the building_id
    missing_count = df[building_id].isna().sum()
    
    # Calculate percentage of missing timestamps
    missing_percentage = (missing_count / total_timestamps) * 100
    
    # Round to 6 decimals as per convention
    missing_percentage = round(missing_percentage, 6)
    
    return {"answer": missing_percentage, "unit": "%"}
