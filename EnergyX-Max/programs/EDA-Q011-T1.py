"""EnergyX-Max program for template EDA-Q011-T1.

For {building_type} buildings at the {site_id} site, what percentage have missing values for year built or floor area?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    site_id = slots.get("site_id")

    # Get list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found at site '{site_id}' with type '{building_type}'.")

    if not building_ids:
        raise ValueError(f"No buildings found matching criteria: site_id='{site_id}', building_type='{building_type}'.")

    # Retrieve metadata for these buildings
    try:
        meta = data.metadata()
    except Exception:
        raise ValueError("Failed to retrieve building metadata.")

    # Filter metadata to only include the selected buildings
    relevant_meta = meta.loc[building_ids]

    if relevant_meta.empty:
        raise ValueError("No valid metadata records found for the specified buildings.")

    # Check for missing values in 'year_built' and 'floor_area_m2'
    # We consider a value missing if it is NaN or an empty string (though pandas usually handles empty strings as non-null, 
    # the prompt says "null or empty strings". In pandas, empty string '' is not NaN. We will check for both).
    
    # Check year_built
    missing_year = relevant_meta['year_built'].isna() | (relevant_meta['year_built'] == '')
    
    # Check floor_area_m2
    missing_area = relevant_meta['floor_area_m2'].isna() | (relevant_meta['floor_area_m2'] == '')

    # Total count of buildings
    total_count = len(relevant_meta)
    
    # Count of buildings with missing values in EITHER field (or both, depending on interpretation of "missing values for year built OR floor area")
    # The question asks: "what percentage have missing values for year built or floor area?"
    # This implies union of the two sets.
    has_missing = missing_year | missing_area
    
    count_missing = has_missing.sum()

    if total_count == 0:
        raise ValueError("Total count of buildings is zero.")

    percentage = (count_missing / total_count) * 100

    # Round to 6 decimals
    percentage = round(percentage, 6)

    return {"answer": percentage, "unit": "%"}
