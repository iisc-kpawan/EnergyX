"""EnergyX-Max program for template CS-042-T1.

What fraction of hourly electricity consumption readings are missing for the {building_id} building?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    
    # Retrieve hourly readings for the specified building
    # The function returns a wide DataFrame where columns are building IDs and values are kWh.
    # We assume the building exists in the dataset. If not, read() might return an empty frame or raise, 
    # but based on conventions, we handle the data retrieval first.
    readings = data.readings(building_ids=[building_id])
    
    if readings.empty:
        raise ValueError(f"No data found for building {building_id}.")

    # Get the total number of hours in the dataset for this building
    total_hours = len(readings)
    
    if total_hours == 0:
        raise ValueError("No valid time range found for the building.")

    # Count missing values (NaN). 
    # Note: The prompt says "null or zero values". However, standard energy data analysis usually treats 0 as a valid reading (no consumption) 
    # unless specified otherwise. The convention section says: "A reading is valid when it is not NaN."
    # It also says "missing readings are defined as null or zero values in the dataset" in the Assumptions.
    # This is a specific instruction for this question. We must count both NaN and 0.
    
    missing_count = (readings.isna().sum()) + (readings == 0).sum()
    
    # Calculate the fraction
    fraction_missing = missing_count / total_hours
    
    # Round to 6 decimals as per convention
    fraction_missing = round(fraction_missing, 6)
    
    return {"answer": fraction_missing, "unit": None}
