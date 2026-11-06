"""EnergyX-Max program for template CS-039-T1.

What percentage of hourly electricity readings are missing for building {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Retrieve readings for the specific building. 
    # We request a wide dataframe with just this building to ensure we get the data correctly.
    # Since start and end are optional, we will fetch all available data for this building.
    try:
        df = data.readings([building_id])
    except Exception:
        raise ValueError(f"No readings found for building {building_id}.")

    if df.empty or building_id not in df.columns:
        raise ValueError(f"Building {building_id} is not present in the dataset.")

    # The column for this building contains hourly kWh values.
    # According to conventions, a reading is valid when it is not NaN.
    # The prompt adds an assumption: "missing readings are defined as null or zero values".
    # So we count both NaN and 0 as missing.
    
    series = df[building_id]
    
    total_hours = len(series)
    if total_hours == 0:
        raise ValueError(f"No data available for building {building_id}.")

    # Count missing (NaN or zero)
    missing_count_nan = series.isna().sum()
    missing_count_zero = (series == 0).sum()
    
    missing_total = missing_count_nan + missing_count_zero
    
    percentage_missing = (missing_total / total_hours) * 100.0

    # Round to 6 decimals as per conventions
    percentage_missing = round(percentage_missing, 6)

    return {"answer": percentage_missing, "unit": "%"}
