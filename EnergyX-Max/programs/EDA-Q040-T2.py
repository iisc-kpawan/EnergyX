"""EnergyX-Max program for template EDA-Q040-T2.

What is the total electricity use for {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    month = slots['month']  # Format: "YYYY-MM"

    # Define the time range for the specified month
    start_ts, end_ts = data.month_range(month)

    # Read the hourly energy readings for the specific building and time period
    # The 'readings' function returns a wide DataFrame where columns are building IDs.
    # We select the column corresponding to our target building_id.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {month}: {str(e)}")

    # Check if the building exists in the dataset (column might be missing if not metered or no data)
    if building_id not in df_readings.columns:
        raise ValueError(f"No electricity use data found for building '{building_id}'.")

    # Extract the series for this building
    series_kwh = df_readings[building_id]

    # Filter for valid readings (not NaN) as per conventions
    valid_series = series_kwh.dropna()

    if valid_series.empty:
        raise ValueError(f"No valid electricity use data found for '{building_id}' in '{month}'.")

    # Calculate the total sum of valid hourly readings
    total_kwh = float(valid_series.sum())

    # Round to 6 decimals as per conventions
    total_kwh_rounded = round(total_kwh, 6)

    return {"answer": total_kwh_rounded, "unit": "kWh"}
