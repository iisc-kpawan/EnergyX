"""EnergyX-Max program for template EDA-Q042-T1.

How does the hourly load profile of {building_id} on a Monday compare to its Sunday profile?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    import pandas as pd
    import numpy as np
    
    building_id = slots['building_id']
    
    # Validate building exists
    if building_id not in data.metadata().index:
        raise ValueError(f"Building {building_id} not found in metadata.")
    
    # Get readings for the specific building
    # We need a wide dataframe to easily compare Monday vs Sunday
    # Since we don't have start/end dates, we will fetch a large enough window 
    # that definitely contains multiple Mondays and Sundays.
    # A safe range is 2 years.
    try:
        readings = data.readings([building_id], start=None, end=None)
    except Exception as e:
        raise ValueError(f"Error fetching readings for {building_id}: {str(e)}")
    
    if building_id not in readings.columns:
        raise ValueError(f"No readings found for building {building_id}.")
    
    col = readings.columns.get_loc(building_id)
    series = readings.iloc[:, col]
    
    # Identify Monday and Sunday indices
    # DatetimeIndex is assumed to be sorted.
    monday_mask = series.index.dayofweek == 1  # Monday is 0, Tuesday 1... wait. 
    # Pandas dayofweek: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6.
    # Correction: 
    monday_mask = series.index.dayofweek == 0
    sunday_mask = series.index.dayofweek == 6
    
    if not monday_mask.any() or not sunday_mask.any():
        raise ValueError("No complete Monday or Sunday data found for the requested building.")
    
    monday_data = series[monday_mask]
    sunday_data = series[sunday_mask]
    
    # Calculate statistics for the record
    stats_mon = {
        "count": int(monday_data.notna().sum()),
        "mean_kwh": round(float(monday_data.mean()), 6),
        "std_kwh": round(float(monday_data.std()), 6),
        "min_kwh": round(float(monday_data.min()), 6),
        "max_kwh": round(float(monday_data.max()), 6)
    }
    
    stats_sun = {
        "count": int(sunday_data.notna().sum()),
        "mean_kwh": round(float(sunday_data.mean()), 6),
        "std_kwh": round(float(sunday_data.std()), 6),
        "min_kwh": round(float(sunday_data.min()), 6),
        "max_kwh": round(float(sunday_data.max()), 6)
    }
    
    # Calculate difference in means
    diff_mean = round(float(stats_mon['mean_kwh'] - stats_sun['mean_kwh']), 6)
    
    return {
        "answer": {
            "Monday": stats_mon,
            "Sunday": stats_sun,
            "Difference (Mon - Sun)": diff_mean
        },
        "unit": "kWh"
    }
