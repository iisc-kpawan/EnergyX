"""EnergyX-Max program for template CS-162-T2.

What is the average hourly electricity consumption for {building_id} at 14:00 in July {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, **slots):
    building_id = slots['building_id']
    year = slots['year']
    month_str = f"{year}-07"
    start, end = data.month_range(month_str)
    readings_df = data.readings([building_id], start=start, end=end)
    if building_id not in readings_df.columns:
        raise ValueError(f"Building {building_id} not found")
    col = readings_df[building_id]
    valid_mask = ~col.isna()
    if not valid_mask.any():
        raise ValueError("No valid readings for the specified period")
    hour_14_mask = (readings_df.index.hour == 14) & valid_mask
    if not hour_14_mask.any():
        raise ValueError("No valid readings at 14:00 in July")
    avg_consumption = col[hour_14_mask].mean()
    return {"answer": round(avg_consumption, 6), "unit": "kWh"}
