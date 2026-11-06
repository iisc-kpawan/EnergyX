"""EnergyX-Max program for template EDA-Q047-T1.

What are the apparent operating hours for {building_id} based on its hourly electricity consumption patterns in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_id, year):
    # Retrieve readings for the specific building and year
    start_ts, end_ts = data.year_range(year)
    readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    
    # Ensure we have a single column for the building
    if building_id not in readings_df.columns:
        raise ValueError(f"No data found for building {building_id} in year {year}.")
    
    consumption_series = readings_df[building_id]
    
    # Filter for valid readings (non-NaN) to calculate median
    valid_consumption = consumption_series.dropna()
    valid_hours_count = len(valid_consumption)
    
    # Check qualifying criteria: at least 90% of the year's hours must be valid
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    expected_hours = 366 if is_leap else 365 * 24
    
    min_valid_hours = expected_hours * 0.9
    
    if valid_hours_count < min_valid_hours:
        raise ValueError(f"Building {building_id} has only {valid_hours_count} valid hours in {year}, below the 90% threshold ({min_valid_hours}).")

    # Calculate median hourly load from valid readings
    median_load = valid_consumption.median()
    
    # Define operating condition: non-zero consumption exceeding the median hourly load
    # Condition: consumption > 0 AND consumption > median_load
    # Effectively: consumption > max(0, median_load)
    threshold = max(0, median_load)
    
    # Create mask for operating hours (valid readings that exceed threshold)
    operating_mask = (consumption_series > threshold) & (~pd.isna(consumption_series))
    
    if not operating_mask.any():
        return {"answer": [], "unit": None}

    # Identify start and end indices of operating periods
    periods = []
    current_start = None
    
    for idx, is_operating in enumerate(operating_mask):
        if is_operating:
            if current_start is None:
                current_start = idx
        else:
            if current_start is not None:
                # End of a period
                periods.append({
                    "start_hour": current_start,
                    "end_hour": idx - 1,
                    "duration_hours": (idx - 1) - current_start + 1
                })
                current_start = None
    
    # Handle the case where the day ends with an operating period
    if current_start is not None:
        periods.append({
            "start_hour": current_start,
            "end_hour": len(consumption_series) - 1,
            "duration_hours": (len(consumption_series) - 1) - current_start + 1
        })
    
    # Format the output record as a list of dictionaries for 'answer'
    result = {
        "answer": periods,
        "unit": None
    }
    
    return result
