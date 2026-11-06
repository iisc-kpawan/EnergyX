"""EnergyX-Max program for template EDA-Q026-T2.

For building {building_id}, what is the total number of hours in 2016 where consumption was zero or below?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Check if the building exists in the metadata
    try:
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building {building_id} not found in metadata.")
    except Exception as e:
        raise ValueError(f"Error accessing metadata: {str(e)}")

    # Get the year range for 2016
    try:
        start, end = data.year_range(2016)
    except Exception as e:
        raise ValueError(f"Error getting year range for 2016: {str(e)}")

    # Read consumption data for the specific building and time period
    try:
        readings_df = data.readings([building_id], start=start, end=end)
    except Exception as e:
        raise ValueError(f"Error reading consumption data: {str(e)}")

    if readings_df.empty or len(readings_df) == 0:
        raise ValueError("No data found for the specified building and year.")

    # Extract the consumption column. The column name should be the building_id itself based on description "one column per building"
    try:
        consumption_series = readings_df[building_id]
    except KeyError:
        raise ValueError(f"Consumption column '{building_id}' not found in readings DataFrame.")

    # Count hours where consumption is zero or below.
    # Assumption states: Zero or negative values indicate missing data or sensor failure.
    # However, the question asks for "total number of hours... where consumption was zero or below".
    # Usually in energy data, 0 might be valid (no load) or invalid (missing). 
    # The prompt says "Zero or negative values indicate missing data or sensor failure."
    # But the question asks to count them. This implies we should count these specific readings as requested by the question text,
    # regardless of the assumption note which explains their physical meaning (likely implying they shouldn't be counted as 'real' consumption, 
    # but the question explicitly asks for the count of such hours).
    # Let's re-read carefully: "what is the total number of hours in 2016 where consumption was zero or below?"
    # And assumption: "Zero or negative values indicate missing data or sensor failure."
    # If the question asks for the count of these specific readings, we count them.
    # If the question implies "valid consumption", we would exclude them. 
    # Given the phrasing "where consumption was zero or below", it refers to the recorded value <= 0.
    
    count_zero_or_below = (consumption_series <= 0).sum()

    return {"answer": int(count_zero_or_below), "unit": None}
