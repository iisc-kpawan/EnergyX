"""EnergyX-Max program for template EDA-Q027-T2.

How many valid readings exist for electricity meter {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    year = slots['year']

    # Get the start and end timestamps for the specified year
    start_ts, end_ts = data.year_range(year)

    # Read the electricity meter readings for the specific building and year range
    # The 'readings' method returns a wide DataFrame where columns are building_ids.
    # We select the column corresponding to our building_id.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {year}: {str(e)}")

    # Check if the building exists in the dataset for this year by checking column existence
    if building_id not in df_readings.columns:
        raise ValueError(f"No data found for building '{building_id}' in the provided dataset.")

    # Extract the series for the specific building
    series = df_readings[building_id]

    # Count valid readings (non-NaN values)
    valid_count = series.notna().sum()

    return {"answer": int(valid_count), "unit": None}
