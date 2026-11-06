"""EnergyX-Max program for template CS-062-T1.

What was the median hourly electricity consumption of {building_id} during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    month = slots['month']  # Format: "YYYY-MM"

    # Get start and end timestamps for the month
    start_ts, end_ts = data.month_range(month)

    # Read hourly consumption data for the specific building
    readings = data.readings([building_id], start=start_ts, end=end_ts)

    # Extract the column for the specific building (it should be the only one or we select by name)
    if building_id not in readings.columns:
        raise ValueError(f"No data found for building {building_id} in month {month}")

    consumption_series = readings[building_id]

    # Calculate median of valid (non-NaN) readings
    median_consumption = consumption_series.median()

    return {"answer": round(float(median_consumption), 6), "unit": "kWh"}
