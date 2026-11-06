"""EnergyX-Max program for template EDA-Q051-T2.

What is the average hourly electricity consumption for {building_type} buildings in {city} during weekends?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, city):
    import numpy as np
    
    # Filter buildings by city and building type
    try:
        building_ids = data.buildings(city=city, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found with building_type='{building_type}' in city='{city}'.")

    if not building_ids:
        raise ValueError(f"No buildings found matching criteria: building_type='{building_type}', city='{city}'.")

    # Get the full year range to ensure we cover all weekends. 
    # We need a range that covers at least one full weekend cycle for every building, 
    # but since we are averaging over the whole dataset's available time, 
    # getting the union of valid hours across all buildings is safer.
    # However, the standard approach for "average hourly consumption" usually implies 
    # using the available data within the context of the question. 
    # Let's get readings for a sufficiently long period to capture weekends.
    # We will fetch readings without specific start/end first to see what's available? 
    # No, we need to define a time range. The problem doesn't specify a year.
    # Usually, in these datasets, we look at the most recent full year or all available data.
    # Let's assume we want to use all available data for these buildings.
    
    # Get readings for all identified buildings. 
    # We need to determine the time range. Let's try to get a wide dataframe.
    # If no start/end is provided, we might hit an error or return empty. 
    # Let's assume we should use the full available history for these buildings.
    # However, data.readings requires start and end usually? The docstring says start=None, end=None.
    # Let's try calling with None to get all data.
    
    try:
        readings = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Error reading data for buildings: {e}")

    if readings.empty:
        raise ValueError("No hourly electricity consumption data found for the specified buildings.")

    # Identify weekend hours (Saturday and Sunday)
    # DatetimeIndex has a 'dayofweek' attribute where 5 is Saturday and 6 is Sunday.
    weekend_mask = readings.index.dayofweek.isin([5, 6])
    
    if not weekend_mask.any():
        raise ValueError("No weekend hours found in the dataset.")

    # Filter readings to weekend hours
    weekend_readings = readings[weekend_mask]

    # Calculate hourly consumption per building (sum of kWh for that hour)
    # The dataframe is already aggregated by hour (one row per hour).
    # We need to sum across buildings for each hour, then average those sums.
    
    # Sum consumption across all buildings for each weekend hour
    total_hourly_consumption = weekend_readings.sum(axis=1)

    if total_hourly_consumption.empty:
        raise ValueError("No valid consumption data found for weekend hours.")

    # Calculate the mean of these hourly totals
    avg_consumption = total_hourly_consumption.mean()

    return {"answer": round(float(avg_consumption), 6), "unit": "kWh"}
