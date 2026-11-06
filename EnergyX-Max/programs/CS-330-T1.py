"""EnergyX-Max program for template CS-330-T1.

For {building_id}, list the hours where hourly electricity consumption exceeds the mean by more than {sd_multiplier} standard deviations.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd

def answer(data, **slots):
    building_id = slots['building_id']
    sd_multiplier = float(slots['sd_multiplier'])
    
    try:
        readings = data.readings([building_id])
        if building_id not in readings.columns:
            raise ValueError(f"Building {building_id} not found.")
        
        series = readings[building_id]
        mean_val = series.mean()
        std_val = series.std()
        
        if np.isnan(mean_val) or np.isnan(std_val):
            raise ValueError("No valid data for the building.")
            
        threshold = mean_val + sd_multiplier * std_val
        hours = series[series > threshold].index.tolist()
        
        return {"answer": sorted(hours), "unit": None}
    except Exception as e:
        raise ValueError(str(e))
