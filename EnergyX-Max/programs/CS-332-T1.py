"""EnergyX-Max program for template CS-332-T1.

Which buildings had hourly electricity consumption exceeding the mean by more than {sd_multiplier} standard deviations on {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    date_str = slots['date']
    sd_mult = float(slots['sd_multiplier'])
    dt = pd.to_datetime(date_str)
    start, end = data.year_range(dt.year)
    if dt < start or dt >= end:
        raise ValueError("Date must be within 2016-2017")
    readings = data.readings(data.buildings(), start=start, end=end)
    target_date = pd.Timestamp(date_str).normalize()
    if target_date not in readings.index:
        return {"answer": [], "unit": None}
    daily_mean = readings.loc[target_date].mean()
    daily_std = readings.loc[target_date].std()
    threshold = daily_mean + sd_mult * daily_std
    valid_buildings = readings.loc[target_date][readings.loc[target_date] > threshold].index.tolist()
    return {"answer": sorted(valid_buildings), "unit": None}
