"""EnergyX-Max program for template EDA-Q075-T2.

What is the average hourly load for {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month) -> dict:
    # Validate inputs and get the specific month range
    start_ts, end_ts = data.month_range(month)
    
    # Get hourly readings for the specified building
    # The read method returns a wide DataFrame where columns are building IDs
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
    
    # Check if the building exists in the dataset at all (though read should handle it)
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")

    # Extract the column for the specific building
    load_series = df_readings[building_id]
    
    # Filter for valid readings (not NaN) as per conventions
    valid_loads = load_series.dropna()
    
    if len(valid_loads) == 0:
        raise ValueError(f"No valid hourly readings found for building '{building_id}' in month '{month}'.")

    # Calculate the average of valid readings
    avg_load = valid_loads.mean()
    
    # Round to 6 decimals as per conventions
    avg_load_rounded = round(float(avg_load), 6)
    
    return {"answer": avg_load_rounded, "unit": "kWh"}
