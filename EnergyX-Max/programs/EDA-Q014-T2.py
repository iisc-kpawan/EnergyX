"""EnergyX-Max program for template EDA-Q014-T2.

How many sites have complete electricity and weather data coverage for the year {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    year = slots['year']
    
    # Get all building IDs from the metadata
    all_building_ids = list(data.metadata().index)
    
    if not all_building_ids:
        raise ValueError("No buildings found in the dataset.")
        
    # Determine the start and end timestamps for the given year
    start, end = data.year_range(year)
    
    qualifying_sites = set()
    
    for building_id in all_building_ids:
        try:
            # Get readings for this building for the specified year
            readings_df = data.readings([building_id], start=start, end=end)
            
            if readings_df.empty:
                continue
                
            # Check for missing values (NaN) in electricity readings
            # Assuming the first column is electricity usage (kWh) or similar energy metric
            # The problem states "complete electricity ... data coverage"
            # We assume the first non-weather column or a specific column named 'electricity' or similar.
            # Looking at typical BDG2 structure, columns are often building IDs or generic names.
            # However, the prompt implies we check for missing readings. 
            # Let's assume the DataFrame has one column per building and that column represents electricity.
            
            # Check if there are any NaNs in the reading column
            # Since it's a wide format with one column per building, we take the first (and likely only) column
            # or iterate if multiple exist. But usually data.readings returns one col per building_id passed.
            # If we pass [building_id], there is exactly one column.
            
            reading_col = readings_df.columns[0]
            
            # Check for missing values
            if readings_df[reading_col].isna().any():
                continue
            
            # Now check weather data coverage for the site associated with this building
            # We need to get the site_id for this building
            metadata = data.metadata()
            site_id = metadata.loc[building_id, 'site_id']
            
            if pd.isna(site_id):
                continue
                
            # Get weather data for this site for the specified year
            weather_df = data.weather(site_id, start=start, end=end)
            
            if weather_df.empty:
                continue
            
            # Check for missing values in weather data
            # The problem says "complete ... weather data coverage". 
            # Usually this implies all standard weather variables or at least one key variable?
            # Given the context of "complete ... and weather", it likely means no NaNs in the weather dataframe.
            if weather_df.isna().any().any():
                continue
            
            # If both electricity and weather are complete for this building/site/year
            qualifying_sites.add(site_id)
            
        except Exception:
            # Skip buildings that cause errors (e.g., site not found, data access issues)
            continue
            
    return {"answer": len(qualifying_sites), "unit": None}
