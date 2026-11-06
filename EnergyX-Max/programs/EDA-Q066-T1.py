"""EnergyX-Max program for template EDA-Q066-T1.

Did {building_id}'s energy consumption spike during hours when outdoor air temperature exceeded 35°C in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    month_str = slots['month']
    
    # Validate inputs
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string")
    if not isinstance(month_str, str):
        raise ValueError("month must be a string in 'YYYY-MM' format")
    
    try:
        start_ts, end_ts = data.month_range(month_str)
    except Exception:
        raise ValueError(f"Invalid month range for '{month_str}'")
    
    # Check if building exists
    if building_id not in data.metadata().index:
        raise ValueError(f"Building '{building_id}' not found in metadata.")
    
    # Get readings for the specific building and month
    try:
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to retrieve readings: {str(e)}")
    
    if readings_df.empty or len(readings_df) == 0:
        return {"answer": "No data available for the specified period.", "unit": None}
    
    # Extract consumption and time index
    consumption = readings_df[building_id]
    timestamps = readings_df.index
    
    # Get weather data for the same period to check temperature
    try:
        weather_df = data.weather(data.metadata().loc[building_id, 'site_id'], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to retrieve weather data: {str(e)}")
    
    if weather_df.empty or len(weather_df) == 0:
        return {"answer": "No weather data available for the specified period.", "unit": None}
    
    # Align timestamps (in case of slight misalignment, though usually they match)
    # We assume the index of readings and weather are aligned by hour.
    # If not, we align them.
    common_index = consumption.index.intersection(weather_df.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps between energy and weather data.")
    
    consumption_aligned = consumption.loc[common_index]
    temp_aligned = weather_df['air_temperature'].loc[common_index]
    
    # Calculate median consumption for the month
    median_consumption = consumption_aligned.median()
    
    # Define spike threshold: 1.5x median
    spike_threshold = 1.5 * median_consumption
    
    # Identify hours where temperature > 35°C
    hot_hours_mask = temp_aligned > 35.0
    
    # Identify hours where consumption is a spike (> 1.5x median)
    spike_hours_mask = consumption_aligned > spike_threshold
    
    # Find intersection: hours that are BOTH hot AND spikes
    both_conditions_mask = hot_hours_mask & spike_hours_mask
    
    # Count occurrences
    count_hot_spikes = both_conditions_mask.sum()
    total_hot_hours = hot_hours_mask.sum()
    
    # Construct the answer text
    if count_hot_spikes == 0:
        ans_text = f"No energy consumption spikes occurred during hours when outdoor air temperature exceeded 35°C in {month_str}."
    elif count_hot_spikes < total_hot_hours:
        ans_text = f"Energy consumption spiked {count_hot_spikes} time(s) out of {total_hot_hours} hot hour(s) when outdoor air temperature exceeded 35°C in {month_str}."
    else:
        ans_text = f"All {total_hot_hours} hours with outdoor air temperature exceeding 35°C in {month_str} saw energy consumption spikes."
    
    return {"answer": ans_text, "unit": None}
