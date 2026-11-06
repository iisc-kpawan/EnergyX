"""EnergyX-Max program for template CS-334-T1.

For {building_id}, list the hourly timestamps where electricity consumption exceeds the mean by more than {sd_multiplier} standard deviations.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_id, sd_multiplier):
    if not isinstance(building_id, str) or not isinstance(sd_multiplier, (int, float)):
        raise ValueError("Invalid slot types")
    try:
        readings = data.readings([building_id])
    except Exception:
        raise ValueError(f"Building {building_id} not found")
    if building_id not in readings.columns:
        raise ValueError(f"Building {building_id} not found")
    col = readings[building_id]
    if col.empty or col.isna().all():
        raise ValueError("No valid data for the building")
    mean_val = col.mean()
    std_val = col.std()
    threshold = mean_val + sd_multiplier * std_val
    mask = col > threshold
    timestamps = readings.index[mask]
    return {"answer": sorted(list(timestamps)), "unit": None}
