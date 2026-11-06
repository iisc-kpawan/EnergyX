"""EnergyX-Max program for template EDA-Q020-T2.

For the {building_id} building, what is the total duration of all gaps in its hourly consumption timeseries exceeding one week?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_id = slots.get('building_id')
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Get the full timeseries for the specific building
    # The readings function returns a wide DataFrame where columns are building_ids.
    # We select the column corresponding to our building_id.
    try:
        df = data.readings(building_ids=[building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
        
    if building_id not in df.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")

    # Ensure we have a valid DatetimeIndex
    if df.index.tz is not None:
        df = df.tz_localize(None)
    
    # Get the full range of data available for this building to determine the theoretical max duration
    # However, the question asks for gaps in "its hourly consumption timeseries". 
    # Usually, this implies the gaps within the observed period.
    # Let's calculate gaps based on the provided data range.
    
    # Identify missing hours (NaN values)
    is_missing = df[building_id].isna()
    
    # Create a boolean series where True indicates a gap
    # We need to find consecutive sequences of True values.
    # A simple way is to group by changes in the missing status.
    
    # Mark the start of a gap: previous was not missing, current is missing
    # Or simply use diff on the cumulative sum of gaps? 
    # Better approach: Identify groups of consecutive NaNs.
    
    # Shift the mask to align with the start of the gap
    # If row i is NaN and row i-1 is not NaN, then row i starts a gap.
    # We can create a series that is 1 if it's the start of a gap, 0 otherwise.
    # Then group by consecutive 1s? No, we want to sum the lengths of consecutive 1s.
    
    # Let's use the fact that a gap starts when current is NaN and previous was not (or start of series).
    # Actually, simpler: 
    # Create a mask of missing values.
    # Calculate the difference between consecutive rows in the mask.
    # A jump from 0 to 1 indicates a start of a gap.
    
    mask = is_missing.values
    # Pad with False at the beginning to handle the first element correctly if it's a gap
    padded_mask = np.concatenate([[False], mask])
    
    # Calculate difference: 1 means transition from non-missing to missing (start of gap)
    # -1 means transition from missing to non-missing (end of gap)
    diff = np.diff(padded_mask, prepend=False) # This logic is slightly flawed with prepend
    
    # Correct logic for finding consecutive runs:
    # 1. Identify indices where a gap starts: mask[i] == True and (i==0 or not mask[i-1])
    starts = []
    ends = []
    
    current_start = -1
    count = 0
    
    for i in range(len(mask)):
        if mask[i]:
            if current_start == -1:
                current_start = i
                count = 1
            else:
                count += 1
        else:
            if current_start != -1:
                # Gap ended at i-1
                duration_hours = count
                if duration_hours > 7 * 24: # Exceeding one week (168 hours)
                    starts.append(current_start)
                    ends.append(i - 1)
                current_start = -1
                count = 0
    
    # Check if the series ends with a gap
    if current_start != -1:
        duration_hours = count
        if duration_hours > 7 * 24:
            starts.append(current_start)
            ends.append(len(mask) - 1)

    total_duration = sum(ends[i] - starts[i] + 1 for i in range(len(starts)))
    
    # Round to 6 decimals as per convention, though it's an integer count of hours usually.
    # The question asks for "total duration", which is a number.
    result = float(round(total_duration, 6))
    
    return {"answer": result, "unit": "hours"}
