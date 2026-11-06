"""EnergyX-Max program for template CS-128-T2.

What is the average hourly electricity consumption for {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    building_id = slots['building_id']
    month = slots['month']
    start, end = data.month_range(month)
    readings_df = data.readings([building_id], start=start, end=end)
    if building_id not in readings_df.columns:
        raise ValueError(f"Building {building_id} not found")
    col = readings_df[building_id]
    valid_values = col.dropna()
    if valid_values.empty:
        raise ValueError("No valid readings for the specified month")
    return {"answer": round(valid_values.mean(), 6), "unit": "kWh"}
