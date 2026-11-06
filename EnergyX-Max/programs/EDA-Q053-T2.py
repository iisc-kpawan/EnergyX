"""EnergyX-Max program for template EDA-Q053-T2.

What is the average hourly electricity consumption (kWh) for building {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month) -> dict:
    # Retrieve hourly readings for the specific building
    # The 'readings' function returns a wide DataFrame where columns are building IDs and rows are timestamps.
    # We select the column corresponding to our building_id.
    try:
        df_readings = data.readings(building_ids=[building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    # Define the time range for the requested month
    start, end = data.month_range(month)

    # Filter the DataFrame to include only rows within the specified month
    # The DatetimeIndex is inclusive of start and exclusive of end.
    mask = (df_readings.index >= start) & (df_readings.index < end)
    df_monthly = df_readings.loc[mask]

    if df_monthly.empty:
        raise ValueError(f"No data found for building {building_id} in month {month}.")

    # Extract the consumption series for this building
    consumption_series = df_monthly[building_id]

    # Filter out NaN values as per conventions ("A reading is valid when it is not NaN")
    valid_consumption = consumption_series.dropna()

    if valid_consumption.empty:
        raise ValueError(f"No valid (non-NaN) readings found for building {building_id} in month {month}.")

    # Compute the average hourly electricity consumption
    avg_consumption = valid_consumption.mean()

    # Round to 6 decimals as per conventions
    avg_consumption_rounded = round(float(avg_consumption), 6)

    return {"answer": avg_consumption_rounded, "unit": "kWh"}
