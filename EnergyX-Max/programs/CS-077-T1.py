"""EnergyX-Max program for template CS-077-T1.

What was the total electricity consumption for {building_id} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    month = slots['month']  # Format: "YYYY-MM"

    # Validate inputs and get time range
    start_ts, end_ts = data.month_range(month)
    
    # Read energy readings for the specific building and month
    # The 'readings' function returns a wide DataFrame with buildings as columns.
    # We select the column corresponding to our building_id.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception:
        raise ValueError(f"No readings found for building {building_id} in month {month}.")

    # Check if the building exists in the metadata (sanity check, though read() usually handles this)
    if building_id not in data.metadata().index:
        raise ValueError(f"Building ID '{building_id}' not found in metadata.")

    # Extract the column for the specific building
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in readings data.")
    
    series_consumption = df_readings[building_id]

    # Calculate total electricity consumption (sum of valid readings)
    # Valid readings are those that are not NaN.
    total_kwh = series_consumption.sum()

    # Round to 6 decimals as per conventions
    total_kwh_rounded = round(float(total_kwh), 6)

    return {"answer": total_kwh_rounded, "unit": "kWh"}
