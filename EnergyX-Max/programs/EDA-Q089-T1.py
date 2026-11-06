"""EnergyX-Max program for template EDA-Q089-T1.

How many {building_type} buildings are available in each {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    country = slots.get("country")

    if not building_type or not country:
        raise ValueError("Missing required slot values: building_type and country.")

    # Filter buildings by country first
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in country '{country}'.")

    if not building_ids:
        return {"answer": {}, "unit": None}

    # Filter by building_type
    # Note: The metadata DataFrame has 'building_type' column.
    # We need to count how many of the filtered building_ids have the specific building_type.
    
    # Get metadata for these buildings
    meta = data.metadata()
    
    # Create a boolean mask for the country (already handled by calling data.buildings, but let's double check logic)
    # Actually, data.buildings returns a list of IDs. We can filter the metadata dataframe directly.
    
    # Filter metadata for the specific country and building type
    filtered_meta = meta[
        (meta["country"] == country) & 
        (meta["building_type"] == building_type)
    ]
    
    if filtered_meta.empty:
        return {"answer": {}, "unit": None}

    # The question asks for counts in each country. 
    # However, the input only provides ONE country. 
    # If the user wants counts per country but only gives one, we just return that one.
    # But wait, the example output says "Table". 
    # If the slots only contain one country, the result is a dict with one key (the country) and value (count).
    # If the question implies aggregating over all countries for a given building type, but the slot restricts to one,
    # we strictly follow the slot. The result should be a record: {country: count}.
    
    count = len(filtered_meta)
    
    return {"answer": {country: count}, "unit": None}
