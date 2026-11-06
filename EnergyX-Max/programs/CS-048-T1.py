"""EnergyX-Max program for template CS-048-T1.

Does meter {building_id} have any duplicated timestamps?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    # Retrieve the readings for the specific building
    # The read method returns a DataFrame with a DatetimeIndex.
    # We assume the index is the timestamp of the reading.
    try:
        df_readings = data.readings(building_ids=[building_id])
    except Exception:
        raise ValueError(f"No data found for building {building_id}")

    if df_readings.empty:
        return {"answer": "No", "unit": None}

    # Check for duplicate timestamps in the index
    # DatetimeIndex has a unique property, but we check duplicates explicitly as per convention
    # or if the underlying data allows multiple entries per hour (though usually index is unique).
    # However, standard pandas read operations might return an index that could have duplicates 
    # if the source data had them, or we are checking the validity of the time series.
    
    has_duplicates = df_readings.index.duplicated().any()

    if has_duplicates:
        return {"answer": "Yes", "unit": None}
    else:
        return {"answer": "No", "unit": None}
