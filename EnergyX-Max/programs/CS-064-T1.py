"""EnergyX-Max program for template CS-064-T1.

What is the peak-to-average ratio of electricity consumption for {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    # Retrieve readings for the specific building and year
    start_ts, end_ts = data.year_range(year)
    
    # Get the wide dataframe of readings. 
    # We pass a list with one element to get that specific building's column.
    df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    
    # Extract the consumption series for the requested building
    consumption_series = df_readings[building_id]
    
    # Filter out NaN values as per conventions (valid readings only)
    valid_consumption = consumption_series.dropna()
    
    if len(valid_consumption) == 0:
        raise ValueError(f"No valid hourly electricity consumption data found for building '{building_id}' in year {year}.")
    
    # Calculate Peak (Maximum)
    peak = valid_consumption.max()
    
    # Calculate Average (Mean)
    average = valid_consumption.mean()
    
    # Handle edge case where average is 0 to avoid division by zero
    if average == 0:
        raise ValueError(f"Average electricity consumption for building '{building_id}' in year {year} is zero. Ratio undefined.")
        
    # Calculate Peak-to-Average Ratio
    ratio = peak / average
    
    # Round to 6 decimals as per conventions
    return {"answer": round(ratio, 6), "unit": None}
