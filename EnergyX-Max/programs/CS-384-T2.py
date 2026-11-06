"""EnergyX-Max program for template CS-384-T2.

Which {top_n} {building_type} buildings have the highest average hourly electricity consumption in {city} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    top_n = slots['top_n']
    building_type = slots['building_type']
    city = slots['city']
    month = slots['month']
    
    bldg_ids = data.buildings(city=city, building_type=building_type)
    if not bldg_ids:
        raise ValueError(f"No {building_type} buildings found in {city}")
    
    start, end = data.month_range(month)
    readings = data.readings(bldg_ids, start=start, end=end)
    
    valid_readings = ~readings.isna().any(axis=1)
    if not valid_readings.any():
        raise ValueError("No valid readings found for the specified period")
    
    monthly_kwh = readings[valid_readings].sum(axis=0)
    rankings = monthly_kwh.sort_values(ascending=False).index
    
    return {"answer": list(rankings[:top_n]), "unit": None}
