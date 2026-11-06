"""EnergyX-Max program for template CS-157-T2.

What is the average hourly electricity consumption for {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    year = slots['year']
    start, end = data.year_range(year)
    readings = data.readings([building_id], start=start, end=end)
    if building_id not in readings.columns:
        raise ValueError(f"Building {building_id} not found")
    col = readings.columns.get_loc(building_id)
    values = readings.iloc[:, col].dropna()
    if len(values) == 0:
        raise ValueError("No valid hourly readings found for the specified year")
    avg_consumption = values.mean()
    return {"answer": round(avg_consumption, 6), "unit": "kWh"}
