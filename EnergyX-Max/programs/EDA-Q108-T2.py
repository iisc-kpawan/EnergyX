"""EnergyX-Max program for template EDA-Q108-T2.

Did {building_id}'s electricity consumption exceed twice its typical daily average on any day in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    import pandas as pd
    import numpy as np
    
    # Get the time range for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Read consumption data for the specific building and year
    readings = data.readings([building_id], start=start_ts, end=end_ts)
    
    if readings.empty:
        raise ValueError(f"No data found for building {building_id} in year {year}.")
    
    # Calculate daily averages
    daily_avg = readings.resample('D').mean()
    
    # Check if there are any days where consumption exceeds twice the typical daily average
    # The typical daily average is the mean of all daily averages
    typical_daily_avg = daily_avg.mean()
    
    threshold = 2 * typical_daily_avg
    
    # Identify days exceeding the threshold
    exceedance_days = daily_avg[daily_avg > threshold]
    
    if len(exceedance_days) > 0:
        return {"answer": f"Yes, on {len(exceedance_days)} day(s).", "unit": None}
    else:
        return {"answer": "No.", "unit": None}
