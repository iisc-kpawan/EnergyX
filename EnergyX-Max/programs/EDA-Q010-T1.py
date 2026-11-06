"""EnergyX-Max program for template EDA-Q010-T1.

What is the percentage of hours where hourly electricity consumption and hourly weather data are both available for the {site_id} site in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, site_id, year):
    import pandas as pd
    
    # Get the time range for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Retrieve building IDs for the given site to ensure correct column names
    building_ids = data.buildings(site_id=site_id)
    
    if not building_ids:
        raise ValueError(f"No buildings found for site {site_id}.")

    # Read hourly electricity consumption for the site(s)
    try:
        readings = data.readings(building_ids, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"No consumption data found for site {site_id} in year {year}.") from e
    
    # Read hourly weather data for the site
    try:
        weather = data.weather(site_id, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"No weather data found for site {site_id} in year {year}.") from e

    # If readings is empty or has no columns, handle it
    if readings.empty or len(readings.columns) == 0:
        raise ValueError("No consumption data available for the requested period.")

    # Align indices to find common hours
    common_index = readings.index.intersection(weather.index)
    
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps found between consumption and weather data.")
    
    # Filter both datasets to the common index
    aligned_consumption = readings.loc[common_index]
    aligned_weather = weather.loc[common_index]
    
    # Identify valid hours where BOTH consumption and weather are not NaN
    # A reading is valid when it is not NaN.
    # Check if any value in the row is NaN for consumption (single column or multiple)
    consumption_valid = ~aligned_consumption.isna().any(axis=1)
    weather_valid = ~aligned_weather.isna().any(axis=1)
    
    valid_mask = consumption_valid & weather_valid
    
    total_hours = len(common_index)
    valid_hours = valid_mask.sum()
    
    if total_hours == 0:
        raise ValueError("No data available for the requested period.")
        
    percentage = (valid_hours / total_hours) * 100
    
    # Round to 6 decimals as per convention
    return {"answer": round(percentage, 6), "unit": "%"}
