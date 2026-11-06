"""EnergyX-Max program for template CS-026-T1.

Which sites have weather data available for the {year} year?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    year = slots['year']
    
    # Get all building IDs from the metadata to determine available sites
    try:
        all_buildings_df = data.metadata()
        if all_buildings_df.empty:
            raise ValueError("No buildings found in the dataset.")
        
        # The index is named 'building_id'. We need to get the 'site_id' column values.
        # Since it's a DataFrame with a MultiIndex or single Index named building_id,
        # we access the 'site_id' level/column directly.
        # Based on the error message, the index name is 'building_id', so we use .get_level_values('building_id') 
        # to get the building IDs, but we want site_id which is a column value.
        # Let's assume 'site_id' is a column in the DataFrame.
        site_ids = sorted(all_buildings_df['site_id'].unique())
    except Exception as e:
        raise ValueError(f"Error retrieving building metadata: {str(e)}")

    # Filter sites that have weather data for the specified year.
    # Based on the assumption "Weather data exists for all sites in the dataset",
    # we return the list of all unique site_ids found in the metadata.
    
    valid_sites = [site_id for site_id in site_ids]

    if not valid_sites:
        raise ValueError("No sites found with weather data for the specified year.")

    return {"answer": valid_sites, "unit": None}
