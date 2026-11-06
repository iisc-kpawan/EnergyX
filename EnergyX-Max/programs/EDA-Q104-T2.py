"""EnergyX-Max program for template EDA-Q104-T2.

Calculate the average hourly electricity consumption for all {building_type} buildings in {city} during {month}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, city, month) -> dict:
    # Filter buildings by city and building type
    try:
        building_ids = data.buildings(city=city, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found in {city} with type {building_type}.")

    if not building_ids:
        raise ValueError(f"No buildings found in {city} with type {building_type}.")

    # Define the time range for the specified month
    start, end = data.month_range(month)

    # Read energy consumption data for the filtered buildings
    try:
        readings = data.readings(building_ids, start=start, end=end)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    # Check if there are any valid readings (non-NaN values) in the dataframe
    if readings.isna().all().all():
        raise ValueError("No valid electricity consumption data found for the specified period.")

    # Calculate hourly sums and means
    # Sum across buildings for each hour
    hourly_totals = readings.sum(axis=1)
    
    # Count non-NaN values per hour to handle missing data correctly
    valid_counts = readings.notna().sum(axis=1)

    # Avoid division by zero if an hour has no data (though unlikely with qualifying buildings)
    # We only compute mean where there is at least one reading. 
    # However, the prompt implies calculating average for the month. 
    # If a specific hour has no data, we might exclude it or fill. 
    # Standard practice: calculate mean of valid hours.
    
    # Calculate hourly averages
    hourly_averages = hourly_totals / valid_counts

    # Filter out hours with zero valid readings (if any) to avoid inf/nan in final average calculation if needed,
    # but usually we just average the existing hours. 
    # Let's ensure we only average hours that have data.
    mask = valid_counts > 0
    hourly_averages_clean = hourly_averages[mask]

    if len(hourly_averages_clean) == 0:
        raise ValueError("No valid hourly consumption data available for the specified month.")

    # Calculate the overall average of these hourly averages
    final_average = hourly_averages_clean.mean()

    return {
        "answer": round(final_average, 6),
        "unit": "kWh"
    }
