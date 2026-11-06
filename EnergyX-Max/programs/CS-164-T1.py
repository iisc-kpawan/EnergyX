"""EnergyX-Max program for template CS-164-T1.

For building {building_id}, does the hourly electricity consumption pattern suggest electric heating is used when outdoor air temperature drops below 10 degrees Celsius?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    try:
        metadata = data.metadata()
        if building_id not in metadata.index:
            raise ValueError("Building ID not found in metadata.")
        
        site_id = metadata.loc[building_id, 'site_id']
        readings_df = data.readings([building_id])
        weather_df = data.weather(site_id)
        
        if readings_df.empty or weather_df.empty:
            raise ValueError("No valid consumption or weather data available.")
        
        merged = pd.merge(readings_df[[building_id]], weather_df[['air_temperature']], left_index=True, right_index=True)
        
        low_temp_mask = merged['air_temperature'] < 10.0
        
        consumption_low = merged.loc[low_temp_mask & (merged[building_id].notna()), building_id]
        consumption_high = merged.loc[(~low_temp_mask) & (merged[building_id] > 0), building_id]
        
        if consumption_low.empty or consumption_high.empty:
            raise ValueError("Insufficient data points to compare consumption at low vs high temperatures.")
        
        mean_low = consumption_low.mean()
        mean_high = consumption_high.mean()
        
        if mean_low > mean_high * 1.2:
            return {"answer": "Yes", "unit": None}
        elif mean_low < mean_high * 0.8:
            return {"answer": "No", "unit": None}
        else:
            return {"answer": "Uncertain", "unit": None}
    except Exception as e:
        raise ValueError(str(e))
