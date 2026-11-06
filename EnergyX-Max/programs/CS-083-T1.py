"""EnergyX-Max program for template CS-083-T1.

What was the total electricity consumption of {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    month = slots['month']  # Format: "YYYY-MM"

    # Validate inputs
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
    if not isinstance(month, str) or len(month) != 7 or not month[4] == '-':
        raise ValueError("month must be in 'YYYY-MM' format.")

    try:
        start_ts, end_ts = data.month_range(month)
    except Exception:
        raise ValueError(f"Invalid month range for '{month}'.")

    # Check if the building exists and is metered
    available_buildings = data.buildings()
    if building_id not in available_buildings:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")

    # Retrieve readings for the specific building and month
    try:
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception:
        raise ValueError("Failed to retrieve energy readings for the specified period.")

    if readings_df.empty or building_id not in readings_df.columns:
        raise ValueError(f"No data found for building '{building_id}' in month '{month}'.")

    # Extract the column for the specific building
    consumption_series = readings_df[building_id]

    # Calculate total electricity consumption (sum of valid, non-NaN readings)
    total_consumption = consumption_series.sum()

    # Round to 6 decimal places as per convention
    total_consumption_rounded = round(float(total_consumption), 6)

    return {"answer": total_consumption_rounded, "unit": "kWh"}
