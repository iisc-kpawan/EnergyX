"""EnergyX-Max program for template CS-056-T1.

For building {building_id}, is an hourly consumption spike of more than {sd_multiplier} standard deviations above the median likely an isolated measurement issue or part of a persistent load pattern?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    sd_multiplier = float(slots['sd_multiplier'])
    
    # Retrieve readings for the specific building
    # We need a wide dataframe to get the column for this building
    try:
        df_readings = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
    
    # Get the column name for the building (usually just the building_id string)
    if building_id not in df_readings.columns:
        raise ValueError(f"Building ID '{building_id}' not found in readings data.")
        
    col_name = building_id
    
    # Extract the series for this building
    consumption_series = df_readings[col_name]
    
    # Calculate median and standard deviation of valid (non-NaN) readings
    median_val = consumption_series.dropna().median()
    std_val = consumption_series.dropna().std()
    
    # Check if standard deviation is zero or NaN to avoid division by zero issues
    if std_val == 0 or std_val != std_val: # NaN check
        return {"answer": "No variation detected in consumption data.", "unit": None}
        
    # Calculate the threshold for a spike
    spike_threshold = median_val + (sd_multiplier * std_val)
    
    # Identify hours where consumption exceeds the threshold
    spike_mask = consumption_series > spike_threshold
    spike_hours_count = spike_mask.sum()
    total_valid_hours = consumption_series.dropna().count()
    
    if total_valid_hours == 0:
        raise ValueError("No valid consumption data found for the building.")

    # Determine if the spikes are isolated or persistent
    # We look at the distribution of hours exceeding the threshold relative to the total valid hours.
    # If a spike is > sd_multiplier std devs above median, it's an outlier by definition.
    # The question asks if it's likely an isolated measurement issue OR part of a persistent pattern.
    
    # Strategy: 
    # 1. Count how many hours exceed the threshold.
    # 2. If the count is very low (e.g., < 5% or < 10 hours in a year), it suggests isolated issues.
    # 3. If the count is significant, it suggests a persistent pattern (or a consistently high load).
    # However, "spike" implies a sudden jump. A persistent high load wouldn't be called a spike usually, 
    # but if the threshold is dynamic (median + k*std), a consistent high load might trigger it every day.
    # Let's assume: 
    # - Isolated issue: Few occurrences (e.g., < 5% of valid hours).
    # - Persistent pattern: Many occurrences (e.g., >= 5% or > 10 hours).
    
    spike_ratio = spike_hours_count / total_valid_hours
    
    # Heuristic thresholds for classification
    # If less than 5% of hours are spikes, consider it isolated issues.
    # Otherwise, consider it part of a persistent pattern (or systematic high usage).
    
    if spike_ratio < 0.05:
        label = "Likely an isolated measurement issue"
    else:
        label = "Part of a persistent load pattern"
        
    return {"answer": label, "unit": None}
