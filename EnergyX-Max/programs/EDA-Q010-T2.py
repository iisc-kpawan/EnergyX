"""EnergyX-Max program for template EDA-Q010-T2.

Using a seasonal naive forecast with {history_days} days of history, what is the mean absolute percentage error for electricity consumption across all buildings in {country} when compared against actual values?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

def answer(data, **slots) -> dict:
    history_days = slots['history_days']
    country = slots['country']
    
    # Get list of building IDs for the specified country
    building_ids = data.buildings(country=country)
    
    if not building_ids:
        raise ValueError(f"No buildings found in country: {country}")
    
    # Get readings for all buildings in the country
    readings = data.readings(building_ids)
    
    if readings.empty:
        raise ValueError("No readings found for the specified buildings.")
    
    # Seasonality period in hours (1 week)
    seasonality_hours = 168
    
    mape_values = []
    
    for bid in building_ids:
        b_readings = readings[bid]
        
        if b_readings.isna().all():
            continue
            
        # Determine the valid range for this building to ensure we have history
        b_last = b_readings.index.max()
        
        # Define test set bounds for this building: last 'history_days' days
        test_start = b_last - pd.Timedelta(days=history_days)
        
        # Ensure we don't go before the start of the dataset for this building
        actual_test_start = max(b_readings.index.min(), test_start)
        
        # Create a mask for the test period
        test_mask = (b_readings.index >= actual_test_start) & (b_readings.index <= b_last)
        test_data = b_readings[test_mask]
        
        if len(test_data) == 0:
            continue
            
        # Generate forecasts using seasonal naive
        # Forecast for hour t is value at t - seasonality_hours.
        # We need to handle the case where t - seasonality_hours is out of bounds or NaN.
        
        forecasts = pd.Series(index=test_data.index, dtype=float)
        valid_forecasts = []
        valid_actuals = []
        
        for idx in test_data.index:
            lag_idx = idx - pd.Timedelta(hours=seasonality_hours)
            if lag_idx in b_readings.index and not pd.isna(b_readings[lag_idx]):
                forecasts[idx] = b_readings[lag_idx]
                valid_forecasts.append(forecasts[idx])
                valid_actuals.append(test_data[idx])
        
        if len(valid_forecasts) == 0:
            continue
            
        # Calculate MAPE for this building
        actuals_array = np.array(valid_actuals)
        forecasts_array = np.array(valid_forecasts)
        
        # Avoid division by zero in MAPE calculation
        mask = actuals_array != 0
        if mask.sum() == 0:
            continue
            
        mape = mean_absolute_percentage_error(actuals_array[mask], forecasts_array[mask])
        mape_values.append(mape)
    
    if not mape_values:
        raise ValueError("No valid data points found to calculate MAPE.")
        
    mean_mape = np.mean(mape_values)
    
    return {"answer": round(mean_mape, 6), "unit": "%"}
