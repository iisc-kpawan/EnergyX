"""EnergyX-Max program for template ADV-17-T2.

What is the average hourly electricity consumption for {building_type} buildings at the {site_id} site in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, site_id, year):
    # Filter buildings by site and type
    building_ids = data.buildings(site_id=site_id, building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No buildings found for site '{site_id}' with type '{building_type}'.")
    
    # Get the start and end timestamps for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Read hourly electricity consumption data for the filtered buildings
    readings_df = data.readings(building_ids, start=start_ts, end=end_ts)
    
    # Check if there are any valid (non-NaN) readings in the dataset
    if readings_df.isna().all().all():
        raise ValueError("No valid electricity consumption data found for the specified criteria.")
    
    # Calculate the sum of valid readings and count of valid hours
    total_consumption = readings_df.sum(axis=0).sum()  # Sum across all buildings and all hours
    valid_hours_count = (~readings_df.isna()).sum().sum()  # Count non-NaN values
    
    if valid_hours_count == 0:
        raise ValueError("No valid hourly data points found to calculate the average.")
    
    # Calculate the average hourly consumption
    average_consumption = total_consumption / valid_hours_count
    
    # Round to 6 decimals as per convention
    return {"answer": round(average_consumption, 6), "unit": "kWh"}
