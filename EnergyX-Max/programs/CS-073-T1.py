"""EnergyX-Max program for template CS-073-T1.

What is the maximum hourly electricity consumption for the {building_id} building in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    # Retrieve the readings for the specific building and year
    # The read method returns a wide DataFrame where columns are building_ids.
    # We need to ensure we get data for the specific building_id requested.
    
    # Define start and end timestamps for the given year
    start_ts, end_ts = data.year_range(year)
    
    # Get readings for the specific building
    # Note: The 'readings' method takes a list of building_ids.
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id} in year {year}: {str(e)}")
    
    # Check if the building exists in the dataset at all (optional safety check based on metadata)
    try:
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building ID '{building_id}' not found in the dataset.")
    except Exception:
        # If metadata access fails for some reason, proceed with what we have or assume it's valid if readings exist
        pass

    # Extract the column for the specific building
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the returned readings data.")
        
    series_consumption = df_readings[building_id]
    
    # Filter out NaN values as per conventions ("A reading is valid when it is not NaN")
    valid_series = series_consumption.dropna()
    
    if len(valid_series) == 0:
        raise ValueError(f"No valid hourly consumption data found for building '{building_id}' in year '{year}'.")

    # Calculate the maximum hourly electricity consumption
    max_consumption = float(valid_series.max())
    
    # Round to 6 decimals as per conventions
    max_consumption = round(max_consumption, 6)
    
    return {"answer": max_consumption, "unit": "kWh"}
