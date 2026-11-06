"""EnergyX-Max program for template EDA-Q025-T2.

What is the total count of buildings in {country} with more than 3 consecutive days of zero electricity readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    country = slots.get('country')
    
    # Get list of building IDs for the specified country
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in country: {country}")

    if not building_ids:
        raise ValueError(f"No buildings found in country: {country}")

    # Define a reasonable date range to cover the dataset (e.g., last 5 years)
    # Since we don't have explicit start/end slots, we try to read a large chunk.
    # If the dataset is huge, we might need to be careful, but usually BDG2 fits in memory for this logic.
    # Let's try to get readings for all buildings without specific date constraints first to see what happens,
    # or assume a standard recent range if needed. However, data.readings allows None for start/end.
    
    try:
        df_readings = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    # Check if the dataframe is empty
    if df_readings.empty or len(df_readings.columns) == 0:
        return {"answer": 0, "unit": None}

    # Ensure datetime index is sorted
    df_readings = df_readings.sort_index()

    count_zero_consecutive_days = 0

    for building_id in df_readings.columns:
        series = df_readings[building_id]
        
        # Identify zero readings. 
        # Note: The problem says "zero electricity readings". In energy data, this usually means 0.0 kWh.
        # NaNs are missing data, not zero consumption. We strictly look for value == 0.
        is_zero = (series == 0)
        
        if not is_zero.any():
            continue

        # Convert boolean series to integer days (1 for zero, 0 for non-zero)
        # We need consecutive days. A day is a 24-hour period. 
        # The data is hourly. Consecutive days means 24 consecutive hours of zeros? 
        # Or simply that the daily total is zero? 
        # "3 consecutive days of zero electricity readings" usually implies the reading for that day (or all readings) are zero.
        # Given hourly data, if ANY hour in a day has non-zero, it's not a zero day.
        # So we check if all 24 hours in a day are zero.
        
        # Resample to daily, taking max or sum? 
        # If any hour is > 0, the day is not zero. So min of the day must be 0? No, sum must be 0.
        # Actually, if consumption is non-negative (which it usually is), sum==0 implies all are 0.
        # Let's use sum to check if the day has any consumption.
        
        daily_sum = series.resample('D').sum()
        
        # A day with zero consumption will have a sum of 0.0 (or close to float precision).
        # We treat values < 1e-9 as zero.
        is_zero_day = daily_sum < 1e-9
        
        if not is_zero_day.any():
            continue
            
        # Find consecutive runs of True in is_zero_day
        # Reset index to get day numbers for easier manipulation if needed, 
        # but resample keeps the DatetimeIndex.
        
        # We can use groupby on changes to find lengths
        diff = is_zero_day.diff().fillna(-1)
        groups = (diff != 0).cumsum()
        
        # Count lengths of each group where value is True
        lengths = []
        for name, group in is_zero_day.groupby(groups):
            if group.all():
                lengths.append(len(group))
        
        # Check if any length >= 3
        if max(lengths, default=0) >= 3:
            count_zero_consecutive_days += 1

    return {"answer": count_zero_consecutive_days, "unit": None}
