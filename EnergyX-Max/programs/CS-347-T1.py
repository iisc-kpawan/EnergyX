"""EnergyX-Max program for template CS-347-T1.

Is the electricity consumption for {building_id} on {date} anomalous using a threshold of {sd_multiplier} standard deviations?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    date_str = slots['date']
    sd_mult = float(slots['sd_multiplier'])
    
    try:
        start, end = data.year_range(2017) # Assuming historical context implies using available full year data or closest complete year if not specified. However, the prompt doesn't specify a reference year for history. Let's assume we use all available data up to the date or a standard recent year. 
        # Re-reading: "historical mean". Usually implies the entire dataset available for that building.
        # Let's fetch readings for the specific building covering the date and potentially more to establish a baseline if needed, 
        # but typically "historical" in these contexts without a specified range means 'all data'.
        # However, to be safe and robust, let's get all readings for this building.
        readings_df = data.readings([building_id])
        
        # Check if building exists in data
        if building_id not in readings_df.columns:
            raise ValueError(f"Building {building_id} not found.")
            
        col = readings_df[building_id]
        
        # Parse the target date
        target_date = pd.to_datetime(date_str)
        
        # Get the reading for the target date (handle timezone if necessary, but DatetimeIndex usually handles it)
        # Ensure the index is datetime
        if not isinstance(readings_df.index, pd.DatetimeIndex):
            readings_df.index = pd.to_datetime(readings_df.index)
            
        target_reading = col.loc[target_date]
        
        if pd.isna(target_reading):
            raise ValueError(f"No reading available for {building_id} on {date_str}.")
            
        # Calculate historical mean and std from ALL valid readings (non-NaN)
        mean_val = col.mean()
        std_val = col.std()
        
        # Check if std is 0 to avoid division by zero issues, though deviation check handles it naturally
        # Anomaly condition: |reading - mean| > sd_mult * std
        deviation = abs(target_reading - mean_val)
        threshold = sd_mult * std_val
        
        is_anomalous = deviation > threshold
        
        if is_anomalous:
            return {"answer": "Anomalous", "unit": None}
        else:
            return {"answer": "Normal", "unit": None}
            
    except Exception as e:
        raise ValueError(str(e))
