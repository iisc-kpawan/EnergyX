"""EnergyX-Max program for template EDA-Q003-T2.

Which {top_n} cities have the most buildings in the dataset?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    import pandas as pd
    
    top_n = slots.get('top_n')
    
    # Get all building IDs from the dataset
    all_building_ids = data.buildings()
    
    if not all_building_ids:
        raise ValueError("No buildings found in the dataset.")
        
    # Count buildings per city using metadata DataFrame directly
    # We need to iterate carefully or use vectorized operations if possible
    # Since metadata is indexed by building_id, we can extract city column
    
    # Create a Series of cities from metadata
    cities_series = data.metadata()['city']
    
    # Filter out None/NaN values
    valid_cities = cities_series.dropna()
    
    # Count occurrences of each city
    city_counts = valid_cities.value_counts()
    
    # Sort by count descending and take top_n
    sorted_cities = city_counts.sort_values(ascending=False).head(top_n)
    top_cities = sorted_cities.index.tolist()
    
    return {"answer": top_cities, "unit": None}
