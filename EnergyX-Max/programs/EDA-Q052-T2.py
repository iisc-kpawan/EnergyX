"""EnergyX-Max program for template EDA-Q052-T2.

What is the average daily electricity consumption of {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    # Retrieve the specific building ID from the slots
    bldg = building_id
    
    # Get the start and end timestamps for the requested year
    start_ts, end_ts = data.year_range(year)
    
    # Read the hourly consumption data for the specific building within the year range
    # The 'readings' function returns a wide DataFrame where columns are building IDs
    readings_df = data.readings([bldg], start=start_ts, end=end_ts)
    
    # Extract the column corresponding to our building
    if bldg not in readings_df.columns:
        raise ValueError(f"Building ID '{bldg}' not found in the dataset.")
        
    consumption_series = readings_df[bldg]
    
    # Filter for valid (non-NaN) readings as per conventions
    valid_readings = consumption_series.dropna()
    
    if len(valid_readings) == 0:
        raise ValueError(f"No valid hourly electricity consumption data found for building '{bldg}' in year {year}.")

    # Calculate the sum of valid readings (total kWh for the year)
    total_kwh = valid_readings.sum()
    
    # Calculate the count of valid hours
    valid_hours_count = len(valid_readings)
    
    # Calculate the average daily consumption
    # Formula: Total kWh / Number of days represented by valid hours
    # Since we have hourly data, number of days = valid_hours_count / 24
    if valid_hours_count == 0:
        raise ValueError("No valid hours found to calculate average.")
        
    days_represented = valid_hours_count / 24.0
    
    if days_represented == 0:
        raise ValueError("Invalid calculation: zero days represented by data.")

    avg_daily_consumption = total_kwh / days_represented
    
    # Round to 6 decimals as per conventions
    avg_daily_consumption = round(avg_daily_consumption, 6)
    
    return {"answer": avg_daily_consumption, "unit": "kWh"}
