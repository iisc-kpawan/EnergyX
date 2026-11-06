"""EnergyX-Max program for template CS-047-T1.

What is the longest consecutive period with no valid readings for building {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_id = slots.get('building_id')
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Get the time range for the building
    try:
        readings = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if readings.empty or len(readings.columns) == 0:
        raise ValueError(f"No readings found for building {building_id}.")

    # Ensure the column exists and is numeric
    col_name = building_id
    if col_name not in readings.columns:
        raise ValueError(f"Column '{building_id}' not found in readings.")
        
    series = readings[col_name]
    
    # Identify NaN positions
    nan_mask = series.isna()
    
    # Calculate consecutive periods of NaNs
    # We can use groupby on the cumulative sum of the boolean mask to identify groups of consecutive NaNs
    if nan_mask.any():
        # Create a group identifier for consecutive NaN blocks
        # Shift the mask down by 1, then cumsum. 
        # When mask is True (NaN), we increment the group id.
        # However, standard trick: diff() on the boolean mask (cast to int) helps identify breaks.
        
        # Let's use a simpler approach: find indices of NaNs, then iterate or use groupby logic.
        nan_indices = nan_mask[series.index].index
        
        if len(nan_indices) == 0:
            return {"answer": 0, "unit": "hours"}

        # Sort indices just in case (readings are usually sorted, but good to be safe)
        nan_indices = nan_indices.sort_values()
        
        # Calculate differences between consecutive NaN indices
        diffs = np.diff(nan_indices.astype(np.int64))
        
        # Consecutive period length is diff + 1. 
        # We need to filter out the first gap (from start of series to first NaN) and last gap (last NaN to end).
        # The diffs represent the gap between index i and index j: j - i.
        # If we have indices [t1, t2, t3], diff is [t2-t1, t3-t2]. These are the lengths of consecutive blocks.
        
        valid_gaps = []
        
        # Check for leading NaNs (from start to first NaN)
        if not nan_indices[0] == series.index[0]:
            valid_gaps.append(nan_indices[0] - series.index[0])
            
        # Check gaps between consecutive NaNs
        current_gap_lengths = diffs[diffs > 0] # Only consider positive differences (consecutive)
        
        # Actually, if indices are t1, t2, t3... then t2-t1 is the length of the block ending at t2? 
        # No. If we have NaN at t1 and t2, the block length is t2 - t1 + 1.
        # diff(t2, t1) = t2 - t1. So length = diff + 1.
        
        if len(nan_indices) > 1:
            diffs = nan_indices.astype(np.int64).diff()
            # Filter out the very first difference which represents the gap from the previous NaN to the current one?
            # No, .diff() on a sorted list of indices gives [idx[1]-idx[0], idx[2]-idx[1], ...]
            # These are exactly the lengths of consecutive sequences minus 1.
            # So length = diff + 1.
            
            gap_lengths = diffs.values + 1
            
            # We must ignore the first element if it corresponds to a break in the sequence? 
            # No, .diff() on [t1, t2] gives [t2-t1]. This is the length of the block between t1 and t2.
            # Wait, if we have NaN at 0, 1, 2. Indices: 0, 1, 2. Diff: 1, 1. Lengths: 2, 2? 
            # Block is 0,1,2 -> length 3. 
            # diff(1,0)=1. diff(2,1)=1. 
            # The logic "diff + 1" works for the segments between them.
            # But we miss the total length if they are all connected?
            # If indices are [0, 1, 2], diffs are [1, 1]. Lengths calculated: [2, 2]. Max is 2. Correct max is 3.
            # The issue is that diff() calculates steps. 
            # Let's reconstruct properly.
            
            # Group consecutive NaN indices
            groups = []
            current_group_start = nan_indices[0]
            current_group_end = nan_indices[0]
            
            for i in range(1, len(nan_indices)):
                prev_idx = nan_indices[i-1]
                curr_idx = nan_indices[i]
                
                if curr_idx == prev_idx + 1:
                    # Consecutive
                    current_group_end = curr_idx
                else:
                    # Gap found
                    groups.append(current_group_end - current_group_start + 1)
                    current_group_start = curr_idx
                    current_group_end = curr_idx
            
            # Append the last group
            groups.append(current_group_end - current_group_start + 1)
            
            if len(groups) == 0:
                return {"answer": 0, "unit": "hours"}
                
            max_gap = max(groups)
            return {"answer": float(max_gap), "unit": "hours"}
        else:
            # Only one NaN reading
            return {"answer": 1.0, "unit": "hours"}
    else:
        # No NaN readings
        return {"answer": 0, "unit": "hours"}
