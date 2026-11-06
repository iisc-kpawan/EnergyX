"""EnergyX-Max program for template CS-297-T2.

Using a seasonal naive forecast with {history_days} days of history, what is the average absolute error in kWh for {building_type} buildings across all sites in {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error

def answer(data, **slots):
    history_days = slots['history_days']
    building_type = slots['building_type']
    country = slots['country']
    
    try:
        building_ids = data.buildings(building_type=building_type, country=country)
        if not building_ids:
            raise ValueError(f"No {building_type} buildings found in {country}")
        
        start, end = data.year_range(2019) # Use a fixed recent year range for consistency if not specified, or use max available. 
        # Since no specific year is given in slots, we assume the latest full year available or a standard one.
        # The prompt implies using history_days of history. We need to define the forecast horizon.
        # Usually, we compare forecast vs actual for the same period. Let's assume we look at the most recent year.
        # To be robust, let's find the latest full year in the data or just use a standard range like 2019-2023.
        # However, without knowing the exact data range, using a fixed recent year (e.g., 2022) might be risky if data ends earlier.
        # Let's try to infer the latest year from metadata or just use a standard approach: 
        # Get readings for all buildings. We need a period where we can forecast.
        # Strategy: Use the last N years of data available.
        
        # Get all readings for these buildings to determine the date range
        all_readings = data.readings(building_ids)
        if all_readings.empty:
            raise ValueError("No readings found")
            
        latest_date = all_readings.index.max()
        # Define a period ending at the latest date. Let's assume we want to evaluate on the last full year or similar.
        # To ensure we have enough history, let's pick a start date that allows for `history_days` of history 
        # before the evaluation period.
        
        # Let's define the evaluation period as the last 365 hours (approx 1 year) ending at latest_date.
        # But we need to handle leap years. Let's just use the last 8760 hours (365 days * 24).
        eval_start = latest_date - pd.Timedelta(days=365)
        eval_end = latest_date
        
        # Ensure we have enough history before eval_start
        history_end = eval_start
        history_start = history_end - pd.Timedelta(hours=history_days)
        
        if history_start < all_readings.index.min():
            raise ValueError("Not enough historical data available")
            
        start_ts = history_start
        end_ts = latest_date + pd.Timedelta(hours=1) # Exclusive
        
        readings = data.readings(building_ids, start=start_ts, end=end_ts)
        
        if readings.empty:
            raise ValueError("No valid readings found in the specified period")
            
        # Filter buildings that have at least some valid data in the history period for forecasting
        # Actually, we need to forecast for the evaluation period using history.
        # Forecast = actual value from `history_days` ago.
        
        errors_list = []
        for bid in readings.columns:
            col = readings[bid]
            if col.isna().all():
                continue
            
            # Check if we have enough valid data points for the forecast logic
            # We need to align history and evaluation periods.
            # The reading at time t is forecasted by reading at t - history_days.
            
            # Create a mask for the evaluation period (last 365 hours)
            eval_mask = readings.index >= eval_start
            
            # For each hour in the evaluation period, get the value from history_days ago
            # We must ensure that the historical value exists and is not NaN
            forecast_values = []
            actual_values = []
            
            for idx in readings[eval_mask].index:
                past_idx = idx - pd.Timedelta(hours=history_days)
                if past_idx in readings.index:
                    hist_val = readings.loc[past_idx, bid]
                    curr_val = readings.loc[idx, bid]
                    
                    if not pd.isna(hist_val) and not pd.isna(curr_val):
                        forecast_values.append(hist_val)
                        actual_values.append(curr_val)
            
            if len(forecast_values) > 0:
                mae = mean_absolute_error(actual_values, forecast_values)
                errors_list.append(mae)
        
        if not errors_list:
            raise ValueError("No valid forecast-error pairs found")
            
        return {"answer": round(sum(errors_list) / len(errors_list), 6), "unit": "kWh"}
    except Exception as e:
        raise ValueError(str(e))
