"""EnergyX-Max program for template EDA-Q045-T2.

Using a seasonal naive forecast with a {history_days}-day history, what is the mean absolute percentage error for predicting hourly load at the {site_id} site for all {building_type} buildings during after-hours (midnight to 4 AM) in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def answer(data, **slots) -> dict:
    history_days = slots['history_days']
    site_id = slots['site_id']
    building_type = slots['building_type']
    year = slots['year']
    
    # Get list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        # Handle cases where arguments might be passed differently or no match found
        # Assuming standard API behavior where None returns all, then we filter manually if needed.
        # However, the spec says buildings(...) takes site_id and building_type directly.
        # If the method doesn't support filtering by both at once in one call (unlikely based on description),
        # we might need to fetch all and filter. Let's assume the method signature supports it as described.
        # If an error occurs or empty list, handle gracefully.
        if not building_ids:
            raise ValueError(f"No buildings found for site_id={site_id} and building_type={building_type}")

    if not building_ids:
        raise ValueError("No buildings found matching the criteria.")

    # Define the time range for the specific year
    start_ts, end_ts = data.year_range(year)
    
    # Read load data for all selected buildings
    try:
        readings_df = data.readings(building_ids, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to read load data: {str(e)}")

    # Get weather data (needed for seasonal naive? Actually, seasonal naive only needs historical load. 
    # But let's check if we need to filter by validity. The problem asks for MAPE on "after-hours".
    # We need to ensure we are using valid readings for the target period and potentially the history period.
    # Standard practice: Forecast uses history (last N days). Target is after-hours in 'year'.
    # We need to construct the seasonal naive forecast.
    
    # Seasonal Naive: For hour h on day d, predict with value from hour h on day (d - history_days)
    # Specifically, for a specific hour of day (e.g., 02:00), we use the same hour from history_days days ago.
    
    # Let's prepare the data
    # readings_df index is DatetimeIndex. Columns are building_ids.
    
    # We need to calculate MAPE for each hour in the target period (after-hours) across all buildings.
    # Target period: Year 'year', hours 00:00 to 04:00.
    # History: last 'history_days' days relative to each forecast point? 
    # Usually, seasonal naive for a specific day uses the same time from 'history_days' days prior.
    # So for a target hour t in year Y, we look at t - history_days * 24 hours.
    
    # Let's create a mask for after-hours (midnight to 4 AM)
    # Midnight is 00:00, 4 AM is 04:00. So hours 0, 1, 2, 3.
    hour_mask = readings_df.index.hour.isin([0, 1, 2, 3])
    
    target_data = readings_df.loc[hour_mask]
    
    if target_data.empty:
        raise ValueError("No after-hours data found in the specified year.")

    # Prepare history for seasonal naive
    # We need to align the forecast values with the actual values.
    # Forecast for time t = Actual value at time (t - history_days * 24 hours)
    
    # Calculate the shift in timestamps
    shift_seconds = history_days * 24 * 3600
    shift_index = target_data.index - pd.Timedelta(seconds=shift_seconds)
    
    # We need to ensure the shifted index exists in the dataframe.
    # Since we read data for the whole year, and history_days is likely small (e.g., 7), 
    # the history might fall into the same year or previous year.
    # The readings_df covers 'year'. If history falls before start_ts, we have NaNs.
    # We should handle missing history gracefully (forecast becomes NaN or we skip).
    
    # Create a mapping of building_id to its series for easier manipulation if needed, 
    # but wide format is fine.
    
    forecast_values = pd.Series(index=target_data.index, dtype=float)
    
    # Iterate over buildings to compute MAPE? Or vectorized?
    # Vectorized approach:
    # For each column (building), shift the series by history_days days and align with target period.
    
    mape_sum = 0.0
    valid_count = 0
    
    for col in readings_df.columns:
        actual_series = readings_df[col].loc[hour_mask]
        
        # Get historical values corresponding to the forecast points
        # We need to map each timestamp in actual_series to (timestamp - history_days)
        # Since the index is DatetimeIndex, we can use reindex or shift logic.
        
        # Create a series of expected historical timestamps
        hist_timestamps = actual_series.index - pd.Timedelta(days=history_days)
        
        # Get values from the original dataframe at these timestamps
        # We must ensure these timestamps are within the range of readings_df
        # If not, they will be NaN.
        predicted_series = readings_df[col].reindex(hist_timestamps)
        
        # Calculate MAPE for this building's after-hours period
        # MAPE = (1/N) * sum(|actual - predicted| / |actual|)
        # Only count where actual != 0 and not NaN
        
        denom = np.abs(actual_series)
        # Avoid division by zero or NaN issues
        valid_mask = (denom > 0) & (~np.isnan(predicted_series)) & (~np.isnan(actual_series))
        
        if valid_mask.any():
            errors = np.abs(actual_series.loc[valid_mask] - predicted_series.loc[valid_mask]) / denom.loc[valid_mask]
            mape_building = errors.mean()
            count_building = valid_mask.sum()
            
            mape_sum += mape_building * count_building
            valid_count += count_building
            
    if valid_count == 0:
        raise ValueError("No valid after-hours data points found for MAPE calculation.")
        
    mean_mape = (mape_sum / valid_count) * 100.0
    
    return {"answer": round(mean_mape, 6), "unit": "%"}
