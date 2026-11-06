"""EnergyX-Max program for template EDA-Q040-T1.

Which month in {year} had the highest electricity use for {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import pandas as pd
    
    building_id = slots['building_id']
    year = slots['year']
    
    # Get the start and end timestamps for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Read the hourly electricity data for the specific building and year
    # The 'readings' function returns a wide DataFrame where columns are building IDs
    readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    
    if building_id not in readings_df.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")
        
    # Extract the time series for the specific building
    # Use 'ME' (month end) instead of deprecated 'M' for resampling
    monthly_usage = readings_df[building_id].resample('ME').sum()
    
    # Find the month with the highest electricity use
    max_month = monthly_usage.idxmax()
    
    return {"answer": str(max_month), "unit": None}
