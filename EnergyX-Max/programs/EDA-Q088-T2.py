"""EnergyX-Max program for template EDA-Q088-T2.

What is the total electricity consumption for buildings in {city} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    city = slots['city']
    month = slots['month']
    
    # Get list of building IDs for the specified city
    building_ids = data.buildings(city=city)
    
    if not building_ids:
        raise ValueError(f"No buildings found in city '{city}'.")
    
    # Define the time range for the month. 
    # The question asks for consumption during {month}, so we use that exact month.
    start, end = data.month_range(month)
    
    # Read the hourly electricity consumption (kWh) for these buildings
    readings = data.readings(building_ids, start=start, end=end)
    
    if readings.empty:
        raise ValueError(f"No reading data available for city '{city}' during month '{month}'.")
    
    # Calculate the total electricity consumption by summing all valid (non-NaN) hourly kWh values
    total_consumption = readings.sum().sum()
    
    return {"answer": round(float(total_consumption), 6), "unit": "kWh"}
