"""EnergyX-Max program for template EDA-Q073-T2.

For building {building_id}, what is the average hourly electricity consumption when outdoor air temperature falls below 10 degrees Celsius?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    
    # Retrieve readings for the specific building
    # We need a wide dataframe to easily filter by building and access weather data if needed, 
    # but the read_weather function takes site_id. Let's first get the site_id from metadata.
    try:
        meta = data.metadata()
        site_id = meta.loc[building_id, 'site_id']
    except KeyError:
        raise ValueError(f"Building {building_id} not found in metadata.")

    # Get the start and end of the available data range to ensure we get all possible hours
    # We can infer the range from the readings or just ask for a very wide range.
    # Let's try to get readings without date constraints first to see what's available, 
    # but the API requires start/end or relies on defaults. 
    # Assuming data.readings handles None gracefully or we need to define a broad range.
    # Let's assume we can request a wide range like 2010-01-01 to 2030-12-31.
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2030-12-31")
    
    # Get electricity readings
    try:
        df_readings = data.readings([building_id], start=start, end=end)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id}: {str(e)}")

    if building_id not in df_readings.columns:
        raise ValueError(f"No electricity consumption data found for building {building_id}.")

    # Get weather data for the same site and time range
    try:
        df_weather = data.weather(site_id, start=start, end=end)
    except Exception as e:
        raise ValueError(f"Could not retrieve weather data for site {site_id}: {str(e)}")

    # Ensure indices align. The readings dataframe has a DatetimeIndex.
    # We need to merge or align the weather data with the readings.
    # df_weather is indexed by hourly timestamp.
    
    # Create a common datetime index
    common_index = pd.DatetimeIndex(df_readings.index)
    
    # Reindex weather to match readings index, filling missing with NaN
    df_weather_aligned = df_weather.reindex(common_index)

    # Extract the relevant columns
    electricity_kwh = df_readings[building_id]
    air_temperature = df_weather_aligned['air_temperature']

    # Filter for hours where air temperature < 10 degrees Celsius
    # Also exclude missing weather readings as per assumptions ("excludes missing weather readings")
    # This implies if temperature is NaN, we drop that hour.
    mask = (air_temperature < 10) & (~air_temperature.isna())
    
    filtered_data = electricity_kwh[mask]

    if filtered_data.empty:
        raise ValueError("No data points found where air temperature was below 10 degrees Celsius.")

    # Calculate the average hourly electricity consumption
    avg_consumption = filtered_data.mean()

    # Round to 6 decimals
    avg_consumption_rounded = round(float(avg_consumption), 6)

    return {"answer": avg_consumption_rounded, "unit": "kWh"}
