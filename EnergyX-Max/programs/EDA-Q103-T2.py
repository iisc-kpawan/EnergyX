"""EnergyX-Max program for template EDA-Q103-T2.

Using a seasonal naive forecast with a one-week history, what is the predicted electricity consumption for {building_id} on {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, date):
    import pandas as pd
    from datetime import timedelta
    
    # Parse the target date
    target_date = pd.to_datetime(date)
    
    # Calculate the start of the history window (one week prior)
    # The problem states "data exists for exactly one week prior", so we take the 7 days before.
    # Since the input is a specific date, we assume the forecast is for that specific hour? 
    # Or daily? The question asks for consumption on {date}. Usually this implies daily total or hourly average.
    # Given "seasonal naive forecast with a one-week history", and standard BDG2 tasks often asking for daily totals 
    # when given a date, let's look at the data structure. 
    # data.readings returns wide DataFrame of hourly kWh.
    # If the question asks for consumption on a date, it likely implies the sum for that day (daily total).
    
    start_date = target_date - timedelta(days=7)
    end_date = target_date + timedelta(days=1) # Exclusive end for reading
    
    # Get readings for the building
    # We need to ensure we get data for the history period and the target date.
    try:
        readings_df = data.readings([building_id], start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for {building_id}: {str(e)}")
    
    if building_id not in readings_df.columns:
        raise ValueError(f"Building ID '{building_id}' not found in data.")
        
    target_col = readings_df[building_id]
    
    # Check if we have exactly one week of history (7 days * 24 hours = 168 hours) prior to the target date.
    # The problem assumes data exists, but let's verify the index range roughly matches expectations 
    # or just proceed with available data if it fits the "one-week history" logic.
    
    # Seasonal Naive Forecast for a specific day:
    # We look at the same day of the week in the past 7 days (or fewer if data is missing, but assumption says exactly one week).
    # Since we need to predict for 'date', we find the corresponding day in the history window.
    
    # Identify the day of the week for the target date
    target_dow = target_date.dayofweek
    
    # Find the index of the last valid reading before the target date that corresponds to the same day of the week
    # History window: start_date to target_date (exclusive)
    history_end = target_date - timedelta(days=1)
    
    # Filter readings for the history period
    history_mask = (readings_df.index >= start_date) & (readings_df.index <= history_end)
    history_values = target_col[history_mask].dropna()
    
    if len(history_values) == 0:
        raise ValueError("No valid historical data found in the one-week prior period.")
        
    # Find the most recent occurrence of the same day of the week before the target date
    # We iterate backwards from the day before the target date to find the matching dow.
    # Since we assume exactly one week history exists, there should be exactly one such day in the 7 days prior.
    
    candidate_date = history_end
    while candidate_date >= start_date:
        if candidate_date.dayofweek == target_dow:
            # Found the corresponding historical day
            # Get the value at this timestamp
            hist_val = target_col.loc[candidate_date]
            if pd.notna(hist_val):
                return {"answer": float(hist_val), "unit": "kWh"}
            else:
                # Should not happen given assumptions, but fallback to next available? 
                # No, seasonal naive specifically looks for the same day of week.
                raise ValueError(f"No valid reading found for {candidate_date.date()} in history.")
        candidate_date -= timedelta(days=1)
        
    raise ValueError("Could not find a corresponding historical day of the week.")
