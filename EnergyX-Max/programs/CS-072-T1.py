"""EnergyX-Max program for template CS-072-T1.

What was the total electricity consumption for all buildings in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    month = slots['month']
    
    # Define the start and end timestamps for the given month
    start, end = data.month_range(month)
    
    # Read hourly consumption data for all buildings in the specified month range
    # Pass an empty list to get all buildings instead of None
    readings = data.readings(building_ids=[], start=start, end=end)
    
    # Sum the electricity consumption across all valid (non-NaN) readings
    total_consumption = readings.sum().sum()
    
    return {"answer": round(float(total_consumption), 6), "unit": "kWh"}
