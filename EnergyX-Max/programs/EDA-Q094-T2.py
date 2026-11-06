"""EnergyX-Max program for template EDA-Q094-T2.

Which {top_n} sites in {country} have the highest average winter electricity intensity (kWh/m²) for {building_type} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    top_n = slots.get("top_n")
    country = slots.get("country")
    building_type = slots.get("building_type")

    # Get list of building IDs matching the criteria
    try:
        building_ids = data.buildings(country=country, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for country='{country}' and building_type='{building_type}'.")

    if not building_ids:
        raise ValueError("No qualifying buildings found.")

    # Define winter months (December, January, February)
    # We need to construct a date range covering all Dec-Jan-Feb combinations across the available data.
    # Since we don't know the exact start/end of the dataset without calling year_range or month_range,
    # we will iterate through years present in the metadata or readings if necessary.
    # However, a robust way is to define a wide range and let the reading function handle NaNs.
    # Let's assume we need to cover multiple years. We can try to get readings for a long span.
    
    # Strategy: 
    # 1. Determine the range of years available in the data by checking metadata or trying to read.
    #    Since we don't have a direct 'get_years' method, we can infer from building year_built + some buffer,
    #    OR simply try to read a very wide range and see what exists. 
    #    Actually, the safest bet for "average winter" across history is to collect all valid winter months.
    #    Let's try to get readings for a span of 30 years (2000-2030) which usually covers most datasets.
    
    import pandas as pd
    
    # Define a wide range to capture all possible winters in the dataset
    start_date = "2000-01-01"
    end_date = "2030-01-01"
    
    try:
        readings_df = data.readings(building_ids, start=start_date, end=end_date)
    except Exception as e:
        # If the range is too wide or invalid, try to narrow it down based on building years?
        # Or just let it fail if no data exists. 
        # Let's assume standard dataset bounds. If this fails, we might need dynamic bounds.
        # But usually, a fixed wide range works for these challenges.
        raise ValueError(f"Could not retrieve readings for the specified period: {e}")

    # Filter for winter months (Dec, Jan, Feb)
    # Create a mask for winter months
    winter_months = [12, 0, 1] # Dec=12, Jan=0, Feb=1
    
    # Identify valid buildings for the calculation
    # A building must have data for at least one winter month to be considered? 
    # The prompt implies "average winter electricity intensity". Usually this means average over all available winters.
    # If a building has no winter data in the dataset, it shouldn't be included or its value is 0/NaN.
    # We will calculate total kWh and floor area for each building across ALL winter months present in the dataframe.
    
    # Prepare to store results: {building_id: (total_kwh, floor_area)}
    building_stats = {}
    
    # Iterate over buildings to sum up winter consumption
    # Note: readings_df is wide format (columns are building_ids)
    for bid in building_ids:
        if bid not in readings_df.columns:
            continue
            
        col_data = readings_df[bid]
        
        # Filter rows where month is in winter list
        winter_mask = col_data.index.month.isin(winter_months)
        winter_kwh = col_data[winter_mask].sum()
        
        if pd.isna(winter_kwh):
            continue
            
        # Get floor area from metadata
        try:
            meta = data.metadata()
            floor_area = meta.loc[bid, 'floor_area_m2']
            if pd.isna(floor_area) or floor_area <= 0:
                continue
        except Exception:
            raise ValueError(f"Floor area not found for building {bid}")
            
        building_stats[bid] = (winter_kwh, floor_area)

    if not building_stats:
        raise ValueError("No winter consumption data found for the specified buildings.")

    # Calculate EUI (kWh/m2)
    euis = []
    for bid, (kwh, area) in building_stats.items():
        eui = kwh / area
        euis.append((bid, eui))

    # Sort by EUI descending and take top_n
    euis.sort(key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        top_building_ids = [x[0] for x in euis[:top_n]]
    else:
        top_building_ids = [x[0] for x in euis]

    # Return as a sorted list of strings (building IDs)
    return {"answer": sorted(top_building_ids), "unit": None}
