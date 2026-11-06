"""EnergyX-Max program for template EDA-Q086-T2.

What is the total hourly electricity consumption for {building_type} buildings in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, month) -> dict:
    import pandas as pd
    
    # Determine the start and end timestamps for the given month
    start, end = data.month_range(month)
    
    # Get a list of building IDs that match the specified building type
    building_ids = data.buildings(building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No buildings found with building type '{building_type}'.")
    
    # Read the hourly consumption data for these buildings within the month range
    # The 'readings' function returns a wide DataFrame where columns are building IDs
    readings = data.readings(building_ids, start=start, end=end)
    
    if readings.empty:
        raise ValueError("No consumption data found for the specified buildings and time period.")
    
    # Sum the hourly consumption across all columns (buildings) for each timestamp
    total_hourly_consumption = readings.sum(axis=1)
    
    # Calculate the sum of these totals for the entire month
    monthly_total = total_hourly_consumption.sum()
    
    return {"answer": round(float(monthly_total), 6), "unit": "kWh"}
