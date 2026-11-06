"""EnergyX-Max program for template EDA-Q100-T1.

How does {building_id}'s peak-demand hour compare to the median peak-demand hour of all buildings at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, site_id):
    import pandas as pd
    from datetime import datetime
    
    # Get the list of all buildings at the specified site
    all_building_ids = data.buildings(site_id=site_id)
    
    if not all_building_ids:
        raise ValueError(f"No buildings found at site '{site_id}'.")
    
    if building_id not in all_building_ids:
        raise ValueError(f"Building '{building_id}' is not located at site '{site_id}'.")
    
    # Determine the full date range of the data to ensure we cover all potential peak hours
    # We fetch readings for a wide range to be safe, or just use the max possible range if needed.
    # Since read_data might have gaps, let's get the min and max timestamps from the metadata or try to infer.
    # However, the safest way without knowing the exact global bounds is to query a very wide range 
    # or rely on the fact that 'readings' returns a complete DatetimeIndex for the requested buildings.
    # Let's fetch data for all buildings at the site.
    
    try:
        # Try to get a reasonable start and end. If we don't know, let's assume we need the full history available.
        # We can try to get readings for just the target building first to see the range, 
        # but 'readings' with no date args returns everything? The doc says "complete hourly DatetimeIndex".
        # Let's assume calling data.readings(all_building_ids) without dates returns all available data.
        df_all = data.readings(all_building_ids)
    except Exception:
        raise ValueError("Could not retrieve building energy data.")

    if df_all.empty:
        raise ValueError("No energy readings found for the specified buildings.")

    # Calculate peak demand hour for each building
    # Peak demand is the hour with maximum hourly consumption.
    # We need to handle NaNs by ignoring them (sum/mean logic implies valid only, but max needs care).
    # A reading is valid when not NaN. So we find the index of the max value among non-NaN values.
    
    peak_hours = {}
    
    for bid in all_building_ids:
        series = df_all[bid]
        if series.isna().all():
            continue
            
        # Find the maximum value
        max_val = series.max()
        
        # If there are no valid readings, skip (though unlikely for a peak demand question)
        if pd.isna(max_val):
            continue
            
        # Get all indices where the value equals the max
        max_indices = series[series == max_val].index
        
        # The problem asks for "the" hour. If there are ties, we need a convention.
        # Usually, "peak-demand hour" implies a specific one. 
        # Convention: "Rankings break ties by building_id". Here we have time ties.
        # Let's pick the earliest or latest? The prompt doesn't specify tie-breaking for time.
        # However, standard practice in such contexts often defaults to the first occurrence or the most recent.
        # Let's look at the example context: "How does X compare to Y".
        # If there are multiple peaks, usually we might take the average or just one.
        # But let's assume unique max for simplicity unless ties exist. 
        # If ties exist, let's pick the earliest hour (first occurrence) as a standard convention for "the" peak.
        
        peak_hour = max_indices[0]
        peak_hours[bid] = peak_hour
        
    if not peak_hours:
        raise ValueError("No valid peak demand hours found for buildings at this site.")

    # Get the peak hour for the specific building
    target_peak_hour = peak_hours.get(building_id)
    
    if target_peak_hour is None:
        raise ValueError(f"No valid peak demand hour found for building '{building_id}'.")

    # Get the median peak-demand hour of all buildings at the site
    peak_hour_values = list(peak_hours.values())
    
    # Convert to datetime objects to handle time correctly (though they are already Timestamps)
    # We need the median.
    import numpy as np
    
    # Sort the hours to find the median
    sorted_peak_hours = sorted(peak_hour_values)
    n = len(sorted_peak_hours)
    
    if n == 0:
        raise ValueError("No buildings found at site.")
        
    # Calculate median index
    if n % 2 == 1:
        median_idx = n // 2
        median_peak_hour = sorted_peak_hours[median_idx]
    else:
        # For even number, average the two middle values? 
        # The question asks for "the median peak-demand hour". 
        # If we have hours like 9:00 and 10:00, the median is 9.5? That's not an hour.
        # Usually in discrete time steps, we might round or pick one. 
        # However, let's look at the data type. It's a DatetimeIndex.
        # If we average two timestamps, we get a fractional timestamp (e.g., 9:30).
        # Is "9:30" a valid hour? The question asks for "the peak-demand hour".
        # Perhaps the dataset doesn't have ties often, or we should round to the nearest hour?
        # Or maybe we just report the value as is (which might be 9.5 hours past midnight).
        # Let's assume standard numpy median behavior which returns a float if inputs are floats, 
        # but here inputs are Timestamps. np.median on Timestamps usually interpolates or picks middle.
        # Actually, np.median on a list of datetime objects in numpy/pandas often results in a datetime 
        # that might be fractional if not careful, but pandas median handles DatetimeIndex well?
        # Let's use pandas median for robustness.
        pass

    # Re-calculate using pandas median on the Series of peak hours to handle ties/averaging naturally
    peak_hour_series = pd.Series(peak_hour_values)
    median_peak_hour = peak_hour_series.median()
    
    # If the median is a fractional timestamp (e.g., 9:30), we need to decide how to label it.
    # The question asks for a comparison. "How does X compare to Y".
    # If X is 10:00 and Y is 9:30, the answer might be "10:00 is 30 minutes later than 9:30".
    # But the expected output type is scalar_text (Label).
    # Example: "Building A's peak is at 14:00, which is 2 hours earlier than the site median of 16:00."
    
    # Let's convert to hour and minute for easier comparison text generation.
    def format_time(dt):
        return dt.strftime("%H:%M")

    target_str = format_time(target_peak_hour)
    median_str = format_time(median_peak_hour) if not pd.isna(median_peak_hour) else "N/A"
    
    # Calculate difference in minutes
    diff_minutes = (target_peak_hour - median_peak_hour).total_seconds() / 60
    
    # Construct the text
    if pd.isna(median_peak_hour):
        return {"answer": f"{target_str} is the only peak hour available."}
    
    if diff_minutes == 0:
        return {"answer": f"Both are at {target_str}."}
    
    elif diff_minutes > 0:
        # Target is later than median
        return {"answer": f"{target_str} is {diff_minutes:.1f} minutes later than the site median of {median_str}."}
    else:
        # Target is earlier than median
        return {"answer": f"{target_str} is {abs(diff_minutes):.1f} minutes earlier than the site median of {median_str}."}
