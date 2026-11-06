"""EnergyX-Max program for template CS-244-T1.

Which {site_id} site exhibits the highest average hourly electricity consumption during July of {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, **slots):
    site_id = slots['site_id']
    year = slots['year']
    buildings = data.buildings(site_id=site_id)
    if not buildings:
        raise ValueError(f"No buildings found for site {site_id}")
    start_ts, end_ts = data.month_range(f"{year}-07")
    readings = data.readings(buildings, start=start_ts, end=end_ts)
    avg_consumption = readings.mean(axis=1)
    max_idx = avg_consumption.idxmax()
    return {"answer": max_idx, "unit": None}
