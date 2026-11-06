"""EnergyX-Max program for template CS-154-T1.

What is the total hourly electricity consumption for {building_id} during hours when the outdoor air temperature exceeds 25 degrees Celsius in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    year = slots['year']
    threshold_temp = 25.0
    
    start_ts, end_ts = data.year_range(year)
    readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    
    if building_id not in readings_df.columns:
        raise ValueError(f"Building {building_id} not found.")
        
    weather_df = data.weather(data.metadata().loc[building_id, 'site_id'], start=start_ts, end=end_ts)
    
    valid_mask = ~readings_df[building_id].isna()
    if not (valid_mask.sum() / len(readings_df) >= 0.9):
        raise ValueError("Building does not meet the 90% data availability threshold.")
        
    temp_series = weather_df['air_temperature']
    high_temp_mask = temp_series > threshold_temp
    
    filtered_consumption = readings_df[building_id][valid_mask & high_temp_mask]
    
    if filtered_consumption.empty:
        return {"answer": 0.0, "unit": "kWh"}
        
    total_kwh = filtered_consumption.sum()
    return {"answer": round(total_kwh, 6), "unit": "kWh"}
