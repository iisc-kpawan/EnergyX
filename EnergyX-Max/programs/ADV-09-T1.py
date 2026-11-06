"""EnergyX-Max program for template ADV-09-T1.

For the {building_id} building at the {site_id} site in {country}, which hourly electricity-consumption observations during {month} have an absolute change from the preceding hour exceeding {threshold_pct}% of the mean hourly consumption?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, site_id, country, month, threshold_pct):
    import pandas as pd
    
    # Retrieve the specific month range (start inclusive, end exclusive)
    start_ts, end_ts = data.month_range(month)
    
    # Read the hourly consumption data for the specified building
    # The readings function returns a wide DataFrame where columns are building IDs
    df_readings = data.readings([building_id], start=start_ts, end=end_ts)
    
    if building_id not in df_readings.columns:
        raise ValueError(f"Building {building_id} not found in the dataset.")
    
    # Extract the time series for the specific building
    series = df_readings[building_id]
    
    # Calculate the mean hourly consumption (ignoring NaNs)
    mean_consumption = series.mean()
    
    if pd.isna(mean_consumption):
        raise ValueError("No valid consumption data found for the specified period.")
    
    # Calculate absolute change from the preceding hour
    # Shift by 1 to get previous hour's value, then subtract and take absolute value
    abs_change = series.diff().abs()
    
    # Filter observations where the absolute change exceeds the threshold percentage of the mean
    # threshold_pct is given as a number (e.g., 25 for 25%), so divide by 100.0
    threshold_value = (threshold_pct / 100.0) * mean_consumption
    
    valid_indices = abs_change[abs_change > threshold_value].index
    
    # Convert indices to datetime objects and format as strings YYYY-MM-DD HH:MM
    # The question asks for "hourly electricity-consumption observations", implying the timestamp of that hour.
    # We will return the string representation of the timestamp.
    result_timestamps = pd.to_datetime(valid_indices).strftime('%Y-%m-%d %H:%M')
    
    # Sort the list of timestamps as required for a set/list output
    sorted_timestamps = sorted(result_timestamps)
    
    return {"answer": sorted_timestamps, "unit": None}
