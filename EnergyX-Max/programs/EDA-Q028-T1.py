"""EnergyX-Max program for template EDA-Q028-T1.

What was the total electricity consumption of {building_id} in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month) -> dict:
    # Validate inputs
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
    if not isinstance(month, str):
        raise ValueError("month must be a string in 'YYYY-MM' format.")
    
    # Check if the building exists in the metadata
    try:
        # We assume data.metadata() returns a DataFrame indexed by building_id
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building '{building_id}' not found in the dataset.")
        
        # Get start and end timestamps for the month
        start, end = data.month_range(month)
        
        # Read the consumption data for the specific building and month
        # readings returns a wide DataFrame where columns are building_ids
        df_readings = data.readings([building_id], start=start, end=end)
        
        if building_id not in df_readings.columns:
            raise ValueError(f"No reading data found for building '{building_id}'.")
            
        # Extract the column for the specific building
        series_consumption = df_readings[building_id]
        
        # Calculate total consumption (sum of valid readings, ignoring NaN)
        total_kwh = series_consumption.sum()
        
        # Round to 6 decimals as per convention
        total_kwh_rounded = round(float(total_kwh), 6)
        
        return {"answer": total_kwh_rounded, "unit": "kWh"}
        
    except Exception as e:
        raise ValueError(f"Error processing request: {str(e)}")
