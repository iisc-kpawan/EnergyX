"""EnergyX-Max program for template EDA-Q070-T1.

What is the average hourly electricity consumption for {building_type} buildings at the {site_id} site during hours when outdoor air temperature exceeds 25C and wind speed is below 3 m/s?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    site_id = slots['site_id']
    
    # Retrieve building IDs for the specific site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        # Fallback if arguments are not supported by the method signature exactly as described
        # Assuming 'site_id' is passed directly to buildings. 
        # If 'building_type' needs to be handled differently based on metadata structure, we adjust.
        # Based on description: data.buildings(site_id=None, city=None, country=None, building_type=None)
        building_ids = data.buildings(site_id=site_id, building_type=building_type)

    if not building_ids:
        raise ValueError(f"No buildings found for site '{site_id}' and type '{building_type}'.")

    # Define the time range (2016-2017)
    start_ts = data.year_range(2016)[0]
    end_ts = data.year_range(2017)[1] # End is exclusive
    
    # Read electricity consumption data
    try:
        readings_df = data.readings(building_ids, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    if readings_df.empty:
        raise ValueError("No energy readings found for the specified buildings and time range.")

    # Read weather data for the site
    try:
        weather_df = data.weather(site_id=site_id, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to read weather data: {str(e)}")

    if weather_df.empty:
        raise ValueError("No weather data found for the specified site and time range.")

    # Ensure indices align (readings index is DatetimeIndex, weather index should be too)
    # Align on timestamp
    common_index = readings_df.index.intersection(weather_df.index)
    
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps between energy and weather data.")

    aligned_readings = readings_df.loc[common_index]
    aligned_weather = weather_df.loc[common_index]

    # Filter based on conditions:
    # 1. Outdoor air temperature > 25C
    # 2. Wind speed < 3 m/s
    # Note: 'air_temperature' and 'wind_speed' are the column names from description.
    
    mask_temp = aligned_weather['air_temperature'] > 25.0
    mask_wind = aligned_weather['wind_speed'] < 3.0
    
    valid_mask = mask_temp & mask_wind

    # Apply mask to readings
    filtered_readings = aligned_readings[valid_mask]

    if filtered_readings.empty:
        raise ValueError("No hours found matching the specified weather conditions (Temp > 25C and Wind < 3 m/s).")

    # Calculate average hourly electricity consumption
    # Sum across all buildings for each hour, then mean. 
    # Or simply mean of all values in the filtered dataframe.
    # The question asks for "average hourly electricity consumption", implying the mean value across all valid hours and buildings.
    
    total_consumption = filtered_readings.sum(axis=1)
    avg_consumption = total_consumption.mean()

    return {"answer": round(float(avg_consumption), 6), "unit": "kWh"}
