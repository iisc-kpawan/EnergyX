"""EnergyX-Max program for template CS-076-T2.

Which {building_type} buildings at the {site_id} site had total electricity consumption exceeding the average for that type during {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    site_id = slots['site_id']
    month = slots['month']
    
    # Get the list of building IDs for the specified type at the specified site
    building_ids = data.buildings(site_id=site_id, building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No {building_type} buildings found at site {site_id}.")
    
    # Define the time range for the month
    start_ts, end_ts = data.month_range(month)
    
    # Read the energy consumption data for these buildings
    readings_df = data.readings(building_ids, start=start_ts, end=end_ts)
    
    # Filter rows to only include the specific month
    # The DatetimeIndex is inclusive of start and exclusive of end. 
    # We need to ensure we are looking at hours within the specific month string provided.
    mask = readings_df.index.month == int(month.split('-')[1])
    month_readings = readings_df[mask]
    
    if month_readings.empty:
        raise ValueError(f"No data found for {month} at site {site_id}.")
    
    # Calculate total consumption per building for the month
    monthly_totals = month_readings.sum(axis=0)
    
    # Calculate the average consumption for this building type at this site
    avg_consumption = monthly_totals.mean()
    
    # Identify buildings exceeding the average
    exceeding_buildings = monthly_totals[monthly_totals > avg_consumption].index.tolist()
    
    # Sort the list of building IDs as per convention
    exceeding_buildings.sort()
    
    return {"answer": exceeding_buildings, "unit": None}
