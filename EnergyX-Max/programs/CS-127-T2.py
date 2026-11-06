"""EnergyX-Max program for template CS-127-T2.

What is the average hourly electricity consumption for all {building_type} buildings in {city} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_type, city, month):
    start, end = data.month_range(month)
    bldgs = data.buildings(city=city, building_type=building_type)
    if not bldgs:
        raise ValueError("No buildings found")
    readings = data.readings(bldgs, start=start, end=end)
    valid_cols = [col for col in readings.columns if not readings[col].isna().all()]
    if not valid_cols:
        raise ValueError("No valid consumption data")
    df = readings[valid_cols]
    total_kwh = df.sum(axis=1)
    total_hours = len(total_kwh)
    if total_hours == 0:
        raise ValueError("No available hours")
    avg_kwh = total_kwh.sum() / total_hours
    return {"answer": round(avg_kwh, 6), "unit": "kWh"}
