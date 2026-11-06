"""EnergyX-Max program for template EDA-Q043-T1.

What is the average hourly electricity consumption for {building_id} over a 24-hour period?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    
    # Retrieve readings for the specific building
    # The function data.readings returns a wide DataFrame where columns are building_ids
    try:
        df_readings = data.readings(building_ids=[building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
    
    # Check if the building exists in the data
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")
    
    # Extract the time series for the specific building
    series = df_readings[building_id]
    
    # Filter out NaN values as per convention ("A reading is valid when it is not NaN")
    valid_series = series.dropna()
    
    if len(valid_series) == 0:
        raise ValueError(f"No valid hourly readings found for building {building_id}.")
    
    # Calculate the average (mean) of the valid readings
    avg_consumption = valid_series.mean()
    
    # Round to 6 decimals as per convention
    avg_consumption_rounded = round(float(avg_consumption), 6)
    
    return {"answer": avg_consumption_rounded, "unit": "kWh"}
