"""EnergyX-Max program for template CS-042-T2.

For all {building_type} buildings at the {site_id} site, what is the average fraction of missing hourly electricity consumption readings per building?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    site_id = slots.get("site_id")

    # Get list of building IDs matching the criteria
    building_ids = data.buildings(site_id=site_id, building_type=building_type)

    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}' at site '{site_id}'.")

    # Read all hourly readings for these buildings. 
    # We assume the default range covers the entire dataset available or we need to fetch a wide range.
    # Since no specific date range is given in the slots, we fetch the full available data.
    # The read method handles start/end=None by returning the full available history.
    readings_df = data.readings(building_ids)

    if readings_df.empty:
        raise ValueError("No reading data found for the specified buildings.")

    # Calculate missing fraction per building
    # readings_df has shape (n_hours, n_buildings). Columns are building IDs.
    # We need to count NaNs per column and divide by total non-NaN rows (or total rows if we consider time steps).
    # Convention: "average fraction of missing ... readings". 
    # Usually, this means (count of missing) / (total hours available for that building).
    # However, the definition of "valid" implies we look at the intersection of valid times or just raw missingness.
    # Let's assume standard missing rate: sum(isna) / len(index) for each column.
    
    n_hours = len(readings_df)
    
    # Calculate missing count per building
    missing_counts = readings_df.isna().sum(axis=0)
    
    # Calculate total possible hours (assuming the dataframe index represents the full time span we are analyzing)
    # If a building has fewer valid hours than others, does it have NaNs for the missing ones? 
    # The docstring says: "NaN where missing". This implies if a reading is missing, it's NaN.
    # So total hours = n_hours.
    
    fractions_missing = missing_counts / n_hours
    
    # Calculate the average of these fractions across all buildings
    avg_fraction = fractions_missing.mean()

    # Round to 6 decimals
    avg_fraction_rounded = round(float(avg_fraction), 6)

    return {"answer": avg_fraction_rounded, "unit": None}
