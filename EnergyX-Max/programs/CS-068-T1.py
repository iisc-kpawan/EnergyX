"""EnergyX-Max program for template CS-068-T1.

What is the 95th-percentile hourly electricity demand for {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    year = int(slots['year'])
    
    # Get the time range for the specific year
    start_ts, end_ts = data.year_range(year)
    
    # Read the hourly consumption data for the specific building
    try:
        df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id} in {year}: {str(e)}")
    
    # Ensure the column exists (handle potential naming variations if necessary, though spec implies direct mapping)
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")
        
    series = df_readings[building_id]
    
    # Filter out missing values (NaN) as per conventions
    valid_series = series.dropna()
    
    if len(valid_series) == 0:
        raise ValueError(f"No valid hourly readings found for {building_id} in {year}.")
    
    # Calculate the 95th percentile
    p95_value = np.percentile(valid_series, 95.0)
    
    # Round to 6 decimals as per conventions
    rounded_p95 = round(p95_value, 6)
    
    return {"answer": rounded_p95, "unit": "kWh"}
