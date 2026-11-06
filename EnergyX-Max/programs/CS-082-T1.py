"""EnergyX-Max program for template CS-082-T1.

What was the peak hourly electricity consumption for building {building_id} on {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, date) -> dict:
    import datetime
    
    # Parse the date string "YYYY-MM-DD"
    parts = date.split('-')
    if len(parts) != 3:
        raise ValueError("Invalid date format")
    
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    
    # Construct the start of the specific day (inclusive) and end of the day (exclusive)
    # We use datetime.datetime for the DatetimeIndex alignment.
    start_of_day = datetime.datetime(year, month, day, 0, 0, 0)
    end_of_day = start_of_day + datetime.timedelta(days=1)
    
    # Fetch readings for the building within the specific day range
    try:
        df_readings = data.readings([building_id], start=start_of_day, end=end_of_day)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id} on {date}: {str(e)}")
    
    # Check if the building column exists in the returned dataframe
    if building_id not in df_readings.columns:
        raise ValueError(f"No consumption data found for building {building_id} on {date}")
    
    col_name = building_id
    
    # Filter for valid (non-NaN) readings within the requested date range.
    # The dataframe index should cover the requested time. 
    # We slice to ensure we only look at the specific day if the query returned a wider range.
    valid_readings = df_readings.loc[(df_readings.index >= start_of_day) & (df_readings.index < end_of_day)]
    
    if valid_readings.empty:
        raise ValueError(f"No valid hourly readings found for building {building_id} on {date}")
    
    # Calculate peak consumption (max of valid readings)
    peak_consumption = float(valid_readings[col_name].dropna().max())
    
    return {"answer": round(peak_consumption, 6), "unit": "kWh"}
