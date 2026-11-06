"""EnergyX-Max program for template ADV-08-T2.

What is the average hourly electricity consumption for a specific {building_id} in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    year = int(slots['year'])
    
    # Retrieve hourly readings for the specific building and year
    start_ts, end_ts = data.year_range(year)
    readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    
    # Extract the column for the specific building
    if building_id not in readings_df.columns:
        raise ValueError(f"Building {building_id} not found in readings.")
        
    consumption_series = readings_df[building_id]
    
    # Calculate average of valid (non-NaN) readings
    avg_consumption = consumption_series.mean()
    
    return {"answer": round(float(avg_consumption), 6), "unit": "kWh"}
