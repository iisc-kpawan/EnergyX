"""EnergyX-Max program for template EDA-Q064-T2.

What is the average daily electricity consumption of {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month) -> dict:
    import pandas as pd
    
    # Validate inputs
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
    if not isinstance(month, str):
        raise ValueError("month must be a string in 'YYYY-MM' format.")
    
    # Check if building exists
    try:
        bldgs = data.buildings()
        if building_id not in bldgs:
            raise ValueError(f"Building '{building_id}' not found in the dataset.")
    except Exception as e:
        raise ValueError(str(e))
    
    # Get start and end timestamps for the month
    try:
        start, end = data.month_range(month)
    except Exception as e:
        raise ValueError(f"Invalid month range for '{month}': {str(e)}")
    
    # Read energy readings for the specific building and month
    try:
        readings_df = data.readings([building_id], start=start, end=end)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")
    
    # Check if there is any valid data (non-NaN)
    if readings_df.empty or readings_df[building_id].isna().all():
        raise ValueError("No valid electricity consumption data found for the specified building and month.")
    
    # Calculate total daily consumption per day
    # Group by date, sum the hourly values for each day
    daily_consumption = readings_df.groupby(readings_df.index.date)[building_id].sum()
    
    # Check if we have at least one day of data
    if daily_consumption.empty:
        raise ValueError("No days with electricity consumption recorded in the specified month.")
    
    # Calculate average daily consumption
    avg_daily_consumption = daily_consumption.mean()
    
    return {"answer": round(float(avg_daily_consumption), 6), "unit": "kWh"}
