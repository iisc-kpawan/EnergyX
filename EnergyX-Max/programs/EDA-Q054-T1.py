"""EnergyX-Max program for template EDA-Q054-T1.

Which {top_n} {building_type} buildings have the highest electricity use per square metre in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    top_n = slots['top_n']
    building_type = slots['building_type']
    year = slots['year']
    
    # Retrieve buildings of the specified type
    building_ids = data.buildings(building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}'.")
    
    # Get start and end timestamps for the year
    start, end = data.year_range(year)
    
    # Filter buildings that have at least 90% of the year's hours valid
    qualifying_building_ids = data.qualifying(building_ids, year, min_share=0.9)
    
    if not qualifying_building_ids:
        raise ValueError(f"No qualifying buildings found for type '{building_type}' in year {year}.")
    
    # Read electricity usage for qualifying buildings
    readings_df = data.readings(qualifying_building_ids, start=start, end=end)
    
    # Calculate total hourly electricity use per building
    total_kwh = readings_df.sum(axis=0)
    
    # Get floor area for each building from metadata
    metadata = data.metadata()
    floor_areas = metadata.loc[qualifying_building_ids, 'floor_area_m2']
    
    # Calculate EUI (kWh/m2) for the year
    eui_series = total_kwh / floor_areas
    
    # Sort by EUI in descending order and select top_n
    sorted_eui = eui_series.sort_values(ascending=False)
    top_buildings = sorted_eui.head(top_n).index.tolist()
    
    return {"answer": top_buildings, "unit": None}
