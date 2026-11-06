"""EnergyX-Max program for template EDA-Q095-T1.

Which sites have their highest hourly electricity consumption exactly at 9:00 local time on a typical weekday in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    year = slots['year']
    
    # Define the start and end of the year
    start_ts, end_ts = data.year_range(year)
    
    # Get all building IDs
    all_building_ids = data.buildings()
    
    if not all_building_ids:
        raise ValueError("No buildings found in the database.")

    # Read hourly readings for all buildings for the entire year
    readings_df = data.readings(all_building_ids, start=start_ts, end=end_ts)
    
    # Filter for weekdays only (Monday=0 to Friday=4). 
    # The DatetimeIndex has a 'dayofweek' attribute.
    valid_readings = readings_df[readings_df.index.dayofweek < 5].copy()
    
    if valid_readings.empty:
        raise ValueError("No weekday data found for the specified year.")

    # We are looking for sites where the MAXIMUM consumption occurs exactly at hour 9 (09:00).
    # Note: The question asks about "sites". Sites are associated with buildings.
    # We need to group by site_id.
    
    # Extract site_id from metadata for each building
    metadata = data.metadata()
    site_map = {bid: sid for bid, row in metadata.iterrows() for sid in [row['site_id']]}
    
    # Group readings by site_id and hour
    # First, ensure we have the correct index levels. 
    # valid_readings is indexed by timestamp, columns are building_ids.
    # We need to pivot or groupby to get site_id and hour.
    
    # Create a series of (site_id, hour) -> consumption
    # Since columns are building_ids, we iterate or use stack/pivot logic carefully.
    # Let's create a long format first for easier grouping if needed, but stacking is efficient.
    
    # Stack converts columns to rows, resulting in index: timestamp, column_name (building_id), value
    long_df = valid_readings.stack().reset_index()
    long_df.columns = ['timestamp', 'building_id', 'consumption']
    
    # Add site_id and hour extracted from timestamp
    long_df['site_id'] = long_df['building_id'].map(site_map)
    long_df['hour'] = long_df['timestamp'].dt.hour
    
    # We only care about hour 9
    hour_9_data = long_df[long_df['hour'] == 9].copy()
    
    if hour_9_data.empty:
        raise ValueError("No data found at hour 9 for the specified year.")

    # For each site, find the maximum consumption across ALL hours (valid weekdays)
    # Then check if that max equals the consumption at hour 9.
    
    # Calculate global max per site from hour_9_data? No, we need max over all hours.
    # Let's get the max consumption per site from the full valid dataset (hour 9 included).
    
    # Re-create long format for all hours to calculate true max per site
    all_long = valid_readings.stack().reset_index()
    all_long.columns = ['timestamp', 'building_id', 'consumption']
    all_long['site_id'] = all_long['building_id'].map(site_map)
    
    # Calculate the maximum consumption for each site across all hours
    site_max_consumption = all_long.groupby('site_id')['consumption'].max().to_dict()
    
    # Filter hour_9_data to only include sites where consumption at 9am equals their global max
    candidates = hour_9_data[hour_9_data['consumption'] == hour_9_data['site_id'].map(site_max_consumption)]
    
    if candidates.empty:
        raise ValueError("No site has its highest hourly consumption exactly at 9:00.")
    
    # Extract unique site IDs and sort them
    result_sites = sorted(candidates['site_id'].unique())
    
    return {"answer": result_sites, "unit": None}
