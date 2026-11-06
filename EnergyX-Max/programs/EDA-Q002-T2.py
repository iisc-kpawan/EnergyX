"""EnergyX-Max program for template EDA-Q002-T2.

How many unique building-use types are available for buildings located in {city}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    city = slots.get("city")
    
    if not city:
        raise ValueError("City slot is required.")
        
    # Get list of building IDs for the specified city
    building_ids = data.buildings(city=city)
    
    if not building_ids:
        raise ValueError(f"No buildings found in city '{city}'.")
    
    # Retrieve metadata for these buildings to get their types
    # We assume data.metadata() returns a DataFrame indexed by building_id
    metadata = data.metadata()
    
    # Filter metadata for the specific building IDs we found
    # Using .loc is safe even if the index doesn't match exactly, but here we know they exist.
    # However, to be robust against potential slight mismatches or just filtering:
    city_buildings_meta = metadata.loc[building_ids]
    
    # Extract the 'building_type' column. 
    # The problem asks for "unique building-use types". 
    # Based on conventions and typical schema, 'building_type' is the primary use type.
    # Sometimes there might be a 'building_subtype', but usually 'building_type' is the main category (e.g., Office, Education).
    # Let's assume 'building_type' is the correct column for "building-use types".
    building_types = city_buildings_meta['building_type']
    
    # Get unique values and sort them to ensure a deterministic order if needed, 
    # though the question asks for a count (numeric), so sorting isn't strictly necessary for the final number.
    unique_types = building_types.dropna().unique()
    
    # Count the unique types
    count = len(unique_types)
    
    return {"answer": int(count), "unit": None}
