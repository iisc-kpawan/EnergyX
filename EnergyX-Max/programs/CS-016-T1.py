"""EnergyX-Max program for template CS-016-T1.

Which countries contain data for {building_type} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    
    if not building_type:
        raise ValueError("Missing building_type slot.")
        
    try:
        # Filter buildings by the specified building type
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found with type '{building_type}'.")

    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}'.")

    # Retrieve metadata for these buildings to get their countries
    metadata = data.metadata()
    
    # Filter the metadata DataFrame to only include the buildings we found
    filtered_metadata = metadata.loc[building_ids]
    
    # Extract the 'country' column and remove NaN values if any
    countries = filtered_metadata["country"].dropna().unique()
    
    # Sort the list of countries alphabetically as per convention for sets/lists in this context
    sorted_countries = sorted(countries.tolist())

    return {"answer": sorted_countries, "unit": None}
