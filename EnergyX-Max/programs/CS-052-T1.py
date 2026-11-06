"""EnergyX-Max program for template CS-052-T1.

What percentage of hourly electricity observations are missing for building {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Retrieve readings for the specific building
    # We fetch a wide dataframe. Since we only need one building, we pass the single ID.
    # To ensure we cover the entire history of the building, we don't specify start/end dates.
    try:
        df = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if df.empty or building_id not in df.columns:
        raise ValueError(f"No data found for building {building_id}.")

    # The problem states: "missing data defined as null or zero values in consumption column."
    # In the context of energy data, 0 kWh usually means no consumption (valid), but the prompt explicitly says
    # to treat zero values as missing. We follow the explicit instruction.
    
    col_name = building_id
    
    # Count total hours and missing hours
    # Total hours is the length of the DatetimeIndex
    total_hours = len(df.index)
    
    if total_hours == 0:
        raise ValueError(f"No time data available for building {building_id}.")

    # Identify missing values based on null OR zero
    # Note: The prompt says "null or zero". 
    # In pandas, isna() handles nulls. We manually check for zeros.
    is_missing = df[col_name].isna() | (df[col_name] == 0)
    
    missing_count = is_missing.sum()
    
    # Calculate percentage
    percentage_missing = (missing_count / total_hours) * 100
    
    # Round to 6 decimals as per convention
    percentage_missing = round(percentage_missing, 6)
    
    return {"answer": percentage_missing, "unit": "%"}
