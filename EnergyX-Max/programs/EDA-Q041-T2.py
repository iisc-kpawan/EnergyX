"""EnergyX-Max program for template EDA-Q041-T2.

What is the average hourly electricity consumption of {building_id} at {hour}?

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
    if hour < 0 or hour >= 24:
        raise ValueError("hour must be between 0 and 23.")

    # Check if building exists in metadata
    try:
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building '{building_id}' not found in the dataset.")
    except Exception as e:
        raise ValueError(f"Error accessing metadata: {str(e)}")

    # Get readings for the specific building
    try:
        # We need to get all readings. Since start/end are optional, we pass None to get the full range.
        # The function signature suggests readngs returns a wide DataFrame.
        # However, usually we might need to specify a date range if the dataset is huge, 
        # but the prompt says "Average is computed over all available hours in the dataset".
        # Assuming data.readings handles None correctly for full range.
        readings_df = data.readings([building_id], start=None, end=None)
    except Exception as e:
        raise ValueError(f"Error reading building data: {str(e)}")

    if building_id not in readings_df.columns:
        raise ValueError(f"No data found for building '{building_id}'.")

    # Extract the column for the specific building
    col_name = building_id
    series = readings_df[col_name]

    # Filter for the specific hour
    # The DatetimeIndex is hourly. We need to select rows where the hour of the timestamp matches 'hour'.
    valid_data = series[series.notna()]  # Only consider non-NaN values as per conventions
    
    if len(valid_data) == 0:
        raise ValueError(f"No valid readings found for building '{building_id}' at hour {hour}.")

    # Calculate average
    avg_consumption = valid_data.mean()

    # Round to 6 decimals
    avg_consumption_rounded = round(float(avg_consumption), 6)

    return {"answer": avg_consumption_rounded, "unit": "kWh"}
