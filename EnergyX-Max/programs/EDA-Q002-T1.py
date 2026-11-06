"""EnergyX-Max program for template EDA-Q002-T1.

How many unique building-use types are available for buildings located in {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    country = slots.get("country")
    
    if not country:
        raise ValueError("Country slot is required.")
        
    # Retrieve building IDs for the specified country
    building_ids = data.buildings(country=country)
    
    if not building_ids:
        raise ValueError(f"No buildings found in {country}.")
    
    # Get metadata for these buildings
    metadata = data.metadata()
    
    # Filter metadata to only include the buildings found
    # Note: metadata is indexed by building_id, so we use loc or boolean indexing
    # Since building_ids is a list of strings (building IDs), we can filter directly.
    filtered_metadata = metadata.loc[building_ids]
    
    # Get unique building-use types (building_type column)
    unique_types = filtered_metadata["building_type"].unique()
    
    # Sort the unique types for consistency, though the question asks for a count
    sorted_types = sorted(unique_types)
    
    # The answer is the count of unique types
    count = len(sorted_types)
    
    return {"answer": count, "unit": None}
