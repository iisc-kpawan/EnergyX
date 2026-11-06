"""EnergyX-Max program for template CS-131-T1.

What is the Energy Use Intensity (EUI) in kWh/m2/year for building {building_id} over the full available period?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    try:
        meta = data.metadata()
        floor_area = meta.loc[building_id, 'floor_area_m2']
        if pd.isna(floor_area) or floor_area == 0:
            raise ValueError("No valid floor area found for the building.")
        
        readings = data.readings([building_id])
        if building_id not in readings.columns:
            raise ValueError("Building ID not found in readings.")
        
        col_data = readings[building_id]
        if col_data.isna().all():
            raise ValueError("No valid consumption data found for the building.")
        
        total_kwh = col_data.dropna().sum()
        eui = total_kwh / floor_area
        
        return {"answer": round(eui, 6), "unit": "kWh/m2/year"}
    except Exception as e:
        raise ValueError(str(e))
