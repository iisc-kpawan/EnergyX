"""EnergyX-Max program for template CS-067-T1.

What was the maximum hourly electricity consumption for {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    # Retrieve the specific building ID from slots
    # The function signature expects **slots, so we extract them. 
    # However, the prompt defines the function as def answer(data, **slots).
    # We will assume the caller passes them as keyword arguments or we access them directly if passed as a dict.
    # Based on standard practice for this template:
    
    # Get the time range for the specific year
    start_ts, end_ts = data.year_range(year)
    
    # Read the hourly consumption data for the specific building
    # The readings function returns a wide DataFrame where columns are building IDs
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {year}: {str(e)}")
    
    # Check if the building exists in the dataset at all (though readins might fail or return empty)
    # If the dataframe is empty or the column doesn't exist, handle it.
    if df_readings.empty or building_id not in df_readings.columns:
        raise ValueError(f"No data found for building {building_id} in year {year}.")
    
    # Extract the series for the specific building
    consumption_series = df_readings[building_id]
    
    # Filter out NaN values as per convention ("A reading is valid when it is not NaN")
    valid_consumption = consumption_series.dropna()
    
    if valid_consumption.empty:
        raise ValueError(f"No valid hourly readings found for building {building_id} in year {year}.")
    
    # Calculate the maximum hourly electricity consumption
    max_consumption = valid_consumption.max()
    
    # Round to 6 decimals as per convention
    return {"answer": round(float(max_consumption), 6), "unit": "kWh"}
