"""EnergyX-Max program for template EDA-Q050-T2.

What is the average hourly electricity consumption for {building_id} at {hour}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    hour = slots['hour']
    
    # Validate inputs
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
    if not isinstance(hour, int):
        raise ValueError("hour must be an integer.")
    if hour < 0 or hour > 23:
        raise ValueError("hour must be between 0 and 23.")

    # Get readings for the specific building
    # We need to fetch all available data for this building to calculate the average over "all available hours"
    try:
        readings_df = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if readings_df.empty or building_id not in readings_df.columns:
        raise ValueError(f"No data found for building {building_id}.")

    # Extract the column for the specific building
    col_name = building_id
    values = readings_df[col_name]

    # Filter for the specific hour
    # The DatetimeIndex is hourly. We select rows where the hour matches.
    mask = values.index.hour == hour
    
    filtered_values = values[mask]

    if filtered_values.empty:
        raise ValueError(f"No data available for building {building_id} at hour {hour}.")

    # Calculate average (mean) of valid readings (non-NaN)
    avg_consumption = filtered_values.mean()

    # Round to 6 decimals as per convention
    avg_consumption = round(float(avg_consumption), 6)

    return {"answer": avg_consumption, "unit": "kWh"}
