"""EnergyX-Max program for template CS-066-T1.

What was the average daily electricity consumption for {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    # Retrieve the time range for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Read the hourly consumption data for the specific building and year
    # The 'readings' function returns a wide DataFrame where columns are building IDs.
    # We select the column corresponding to our target building_id.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {year}: {str(e)}")
    
    # Check if the building exists in the dataset at all (though read() might handle it gracefully or return empty)
    # If the column doesn't exist, data.readings usually returns a DataFrame with that column.
    # However, we need to ensure we are summing valid hours.
    
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")

    # Extract the series for this building
    consumption_series = df_readings[building_id]
    
    # Calculate daily sums
    # The index is a DatetimeIndex. We group by date (floor to day) and sum valid (non-NaN) readings.
    daily_consumption = consumption_series.resample('D').sum()
    
    if len(daily_consumption) == 0:
        raise ValueError(f"No valid hourly data found for {building_id} in {year}.")

    # Calculate the average of these daily sums
    avg_daily_consumption = daily_consumption.mean()
    
    return {"answer": round(float(avg_daily_consumption), 6), "unit": "kWh"}
