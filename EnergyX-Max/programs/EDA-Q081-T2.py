"""EnergyX-Max program for template EDA-Q081-T2.

What is the total hourly electricity consumption of {building_type} buildings in {city} for {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, city, month):
    # Filter buildings by city and building type
    try:
        building_ids = data.buildings(city=city, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found in {city} with type {building_type}.")

    if not building_ids:
        raise ValueError(f"No buildings found in {city} with type {building_type}.")

    # Define the time range for the specified month
    start, end = data.month_range(month)

    # Read hourly electricity consumption data for the filtered buildings
    try:
        readings_df = data.readings(building_ids, start=start, end=end)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    if readings_df.empty:
        raise ValueError("No valid hourly readings found for the specified period.")

    # Sum valid (non-NaN) readings across all buildings for the month
    total_consumption = readings_df.sum(axis=1).dropna().sum()

    return {"answer": round(float(total_consumption), 6), "unit": "kWh"}
