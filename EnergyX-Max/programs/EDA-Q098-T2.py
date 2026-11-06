"""EnergyX-Max program for template EDA-Q098-T2.

What is the average hourly electricity consumption for {building_type} buildings at the {site_id} site during July 2016?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import pandas as pd
    
    # Extract slots
    building_type = slots['building_type']
    site_id = slots['site_id']
    
    # Define the month range for July 2016
    start_date = "2016-07-01"
    end_date = "2016-08-01" # Exclusive end date
    
    # Get building IDs matching the criteria
    # Filter by site_id first
    candidate_building_ids = data.buildings(site_id=site_id)
    
    if not candidate_building_ids:
        raise ValueError(f"No buildings found at site '{site_id}'.")
        
    # Filter by building_type
    # Note: The metadata has 'building_type' and 'building_subtype'. 
    # The question asks for 'building_type'. We assume the slot value matches the 'building_type' column.
    filtered_building_ids = [bid for bid in candidate_building_ids if data.metadata().loc[bid, 'building_type'] == building_type]
    
    if not filtered_building_ids:
        raise ValueError(f"No buildings of type '{building_type}' found at site '{site_id}'.")
        
    # Read the energy readings for the specific period and buildings
    try:
        df = data.readings(filtered_building_ids, start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Error reading data: {str(e)}")
        
    # Check if there is any valid data (non-NaN values)
    if df.empty or df.isna().all().all():
        raise ValueError("No valid electricity consumption data found for the specified criteria.")
        
    # Calculate total consumption and total hours
    total_consumption = df.sum(axis=1).sum()
    
    # Count total valid hours across all buildings in the dataframe
    # We need to count every hour that has a value for any building? 
    # Or average per hour across buildings? 
    # "Average hourly electricity consumption ... during July 2016" usually implies:
    # Sum of all kWh consumed by all qualifying buildings / Total number of hours in the period.
    # Let's assume we sum all valid readings and divide by the total number of hours in the month (July has 31 days).
    
    # Total hours in July 2016
    total_hours = 31 * 24
    
    # Calculate average
    if total_consumption != 0:
        avg_hourly_consumption = total_consumption / total_hours
    else:
        avg_hourly_consumption = 0.0
        
    return {"answer": round(avg_hourly_consumption, 6), "unit": "kWh"}
