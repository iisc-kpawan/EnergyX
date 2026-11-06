"""EnergyX-Max program for template EDA-Q017-T1.

What is the longest continuous missing period for electricity meter readings for {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Get the readings for the specific building
    # We fetch all data without time limits to ensure we capture the full history
    try:
        df = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("No readings found for the specified building.")

    # Ensure we are looking at the correct column (the building ID itself is the column name)
    if building_id not in df.columns:
        raise ValueError(f"Column '{building_id}' not found in readings data.")

    col = df[building_id]
    
    # Identify missing periods. A reading is valid if it is not NaN.
    # We create a boolean mask where True means valid, False means missing.
    is_valid = ~col.isna()
    
    # Calculate the difference between consecutive timestamps to find gaps in hours.
    # The index is DatetimeIndex with hourly frequency.
    time_diffs = df.index.to_series().diff()
    
    # A gap exists if the time difference is greater than 1 hour (or exactly 1 hour if data is not strictly hourly, 
    # but BDG2 is typically hourly). Let's assume strict hourly.
    # If diff > timedelta(hours=1), it means there was a missing hour(s).
    # However, simply checking for NaNs in the series is more direct for "consecutive hours with no valid readings".
    
    # Let's reconstruct the timeline of validity.
    # We want to find the longest sequence of False values.
    
    # Convert boolean mask to integer (1 for valid, 0 for missing)
    status = is_valid.astype(int)
    
    # To find consecutive missing periods, we can look at the cumulative sum of gaps or iterate.
    # A simple way: identify indices where status changes from 1 to 0 (start of gap) and 0 to 1 (end of gap).
    # Or simply use numpy logic on the boolean array.
    
    # Let's find runs of zeros.
    # Pad with a 1 at the start to handle gaps starting at t=0 correctly if needed, 
    # though usually we just look at the data range provided.
    
    # Method: Find indices where status is False. Group consecutive indices and calculate lengths.
    missing_indices = col[col.isna()].index
    
    if len(missing_indices) == 0:
        return {"answer": 0, "unit": "hours"}

    # Sort just in case (readings should be sorted, but good practice)
    missing_indices = sorted(missing_indices)
    
    max_gap = 0
    current_gap = 0
    
    # We need to check the gap between the first valid reading and the start of the data if the first is missing?
    # The question asks for "continuous missing period". Usually implies within the dataset's time range.
    # If the dataset starts with missing values, that counts.
    
    # Let's iterate through the boolean series directly.
    # We can use numpy to find runs of False.
    import numpy as np
    
    # Create a list of lengths of consecutive False sequences
    # One trick: concatenate [True] + status + [True] then find indices where value changes? 
    # Easier: iterate.
    
    prev_val = True # Assume before start is valid to ignore leading gaps if desired? 
    # Actually, standard interpretation: count consecutive NaNs in the provided dataset.
    # If the dataset starts with NaNs, that is a missing period.
    
    # Let's use a robust iteration over the boolean array
    vals = status.values
    
    current_run = 0
    max_run = 0
    
    for i in range(len(vals)):
        if vals[i] == 0: # Missing
            current_run += 1
        else: # Valid
            if current_run > max_run:
                max_run = current_run
            current_run = 0
            
    # Check trailing run
    if current_run > max_run:
        max_run = current_run
        
    return {"answer": float(max_run), "unit": "hours"}
