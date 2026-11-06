"""EnergyX-Max program for template ADV-02-T2.

Using a seasonal naive forecast with a {history_days}-day history, what is the MAPE for electricity consumption at the {site_id} site from {date} to {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

def answer(data, **slots) -> dict:
    history_days = slots['history_days']
    site_id = slots['site_id']
    date_str = slots['date']
    
    # Parse the single date provided. The question asks for forecast from {date} to {date}.
    # This implies a specific point in time or a single day evaluation.
    # Given "seasonal period is 24 hours", we interpret this as forecasting for the specific hour(s) 
    # on the given date, using history from 'history_days' prior to that date.
    
    try:
        target_date = pd.to_datetime(date_str)
    except Exception:
        raise ValueError(f"Invalid date format: {date_str}")

    # Get building IDs for the site
    try:
        building_ids = data.buildings(site_id=site_id)
    except Exception as e:
        raise ValueError(f"No buildings found for site_id: {site_id}. Error: {str(e)}")
    
    if not building_ids:
        raise ValueError(f"No buildings found for site_id: {site_id}")

    # Define the target period. Since start == end, we assume the user wants the MAPE 
    # calculated for the 24-hour period ending at the given date (or starting).
    # Standard interpretation for "from date to date" in hourly data with a specific date string:
    # We evaluate the forecast for the hours on that specific date.
    # Let's assume the target period is the 24 hours of the given date.
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1) # Exclusive end
    
    # Calculate history window needed to generate forecasts for the target period.
    # We need data up to (target_start - 24 hours) to forecast target_start.
    # So we need history ending at target_start.
    # The slot 'history_days' defines how much history we have/use.
    # We will fetch data from (start_date - history_days) to end_date.
    
    read_start = start_date - timedelta(days=history_days)
    read_end = end_date
    
    readings_df = data.readings(building_ids, start=read_start, end=read_end)
    
    if readings_df.empty:
        raise ValueError("No electricity consumption data found for the specified site and date range.")

    # Prepare data for each building
    mape_values = []
    
    for bid in building_ids:
        col_name = f"{bid}"
        if col_name not in readings_df.columns:
            continue
            
        series = readings_df[col_name]
        
        # Ensure index is DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
            
        # Filter for the target period (the specific date provided)
        mask = (series.index >= start_date) & (series.index < end_date)
        target_series = series[mask]
        
        if target_series.empty:
            continue
            
        # Generate Seasonal Naive Forecast
        # Assumption: Seasonal period is 24 hours.
        # Forecast[t] = Actual[t - 24 hours].
        
        forecast_values = []
        actual_values = []
        
        for idx, val in target_series.items():
            if pd.isna(val):
                continue
            
            # Find the corresponding historical value: idx - 24 hours
            hist_idx = idx - timedelta(hours=24)
            
            # Check if this historical index exists and is valid
            if hist_idx not in series.index:
                continue
            
            hist_val = series[hist_idx]
            
            if pd.isna(hist_val):
                continue
                
            forecast_values.append(hist_val)
            actual_values.append(val)
        
        if len(forecast_values) == 0:
            continue
            
        # Calculate MAPE
        # MAPE = (1/N) * sum(|Actual - Forecast| / Actual)
        errors = np.abs(np.array(actual_values) - np.array(forecast_values))
        actuals_arr = np.array(actual_values)
        
        # Filter out zeros in actuals to avoid div by zero
        valid_mask = (actuals_arr != 0) & ~np.isnan(actuals_arr)
        if not np.any(valid_mask):
            continue
            
        mape = np.mean(errors[valid_mask] / actuals_arr[valid_mask])
        mape_values.append(mape)

    if not mape_values:
        raise ValueError("Could not compute MAPE due to insufficient valid data points for forecasting.")
        
    total_mape = np.mean(mape_values)
    
    return {"answer": round(total_mape, 6), "unit": "%"}
