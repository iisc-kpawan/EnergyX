"""EnergyX-Max program for template CS-076-T1.

What was the total electricity consumption for {building_id} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month) -> dict:
    # Validate inputs and get timestamps
    start_ts, end_ts = data.month_range(month)
    
    # Read energy readings for the specific building and time range
    # The 'readings' function returns a wide DataFrame where columns are building_ids.
    # We select the column corresponding to our building_id.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {month}: {str(e)}")
    
    # Check if the building exists in the dataset (column might be missing if not metered or filtered out internally)
    if building_id not in df_readings.columns:
        raise ValueError(f"Building {building_id} not found in the dataset.")

    # Extract the column for the specific building
    series_consumption = df_readings[building_id]
    
    # Calculate total consumption (sum of valid readings, ignoring NaNs)
    total_kwh = series_consumption.sum()
    
    # Round to 6 decimals as per convention
    total_kwh_rounded = round(float(total_kwh), 6)
    
    return {"answer": total_kwh_rounded, "unit": "kWh"}
