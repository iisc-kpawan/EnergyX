"""EnergyX-Max program for template ADV-18-T2.

Which of the top {top_n} {building_type} buildings at the {site_id} site has the highest average hourly electricity consumption in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    top_n = slots.get("top_n")
    building_type = slots.get("building_type")
    site_id = slots.get("site_id")
    year = slots.get("year")
    
    # Retrieve the list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found matching site_id='{site_id}' and building_type='{building_type}'.")

    if not building_ids:
        raise ValueError(f"No buildings found for site '{site_id}' with type '{building_type}'. Cannot compute top {top_n}.")

    # Filter to the specific year range
    start_ts, end_ts = data.year_range(year)
    
    # Get readings for these buildings in the specified year
    try:
        df_readings = data.readings(building_ids, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Error retrieving readings: {str(e)}")

    if df_readings.empty:
        raise ValueError("No valid hourly readings found for the specified buildings and year.")

    # Calculate average hourly consumption per building for the year
    # We sum the kWh (assuming column values are already in kWh as per description) 
    # and divide by the number of non-NaN entries to get the average.
    avg_consumption = df_readings.mean(axis=0)
    
    # Sort by average consumption descending and take top_n
    sorted_buildings = avg_consumption.sort_values(ascending=False).head(top_n)

    if len(sorted_buildings) == 0:
        raise ValueError("No buildings selected after filtering.")

    # Prepare the record output
    result_record = {}
    
    for bid, avg_val in sorted_buildings.items():
        # Round to 6 decimals as per convention
        rounded_avg = round(float(avg_val), 6)
        result_record[str(bid)] = rounded_avg

    return {"answer": result_record, "unit": "kWh"}
