"""EnergyX-Max program for template CS-013-T2.

Using a seasonal naive forecast with {history_days} days history, what is the average absolute error for all {building_type} buildings at the {site_id} site when predicting consumption {horizon_hours} hours ahead?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def answer(data, **slots) -> dict:
    history_days = slots['history_days']
    building_type = slots['building_type']
    site_id = slots['site_id']
    horizon_hours = slots['horizon_hours']

    # Get list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found at site '{site_id}' with type '{building_type}'.")

    if not building_ids:
        raise ValueError(f"No buildings found at site '{site_id}' with type '{building_type}'.")

    # Get readings for all buildings in the list to find the max date
    try:
        readings_df = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for buildings: {e}")

    if readings_df.empty:
        raise ValueError("No readings found for the specified buildings.")

    # Determine the latest date in the dataset to set our end point
    max_date = readings_df.index.max()
    # Set start to 2 years before max_date to ensure we have enough history for training and testing
    min_date = max_date - pd.Timedelta(days=730)
    
    # Ensure we have enough data points. If not, adjust.
    if (max_date - min_date).days < history_days + horizon_hours:
        raise ValueError(f"Not enough historical data available at site '{site_id}' to perform the forecast.")

    start_ts = pd.Timestamp(min_date)
    end_ts = pd.Timestamp(max_date) + pd.Timedelta(days=1) # Exclusive end
    
    # Re-fetch readings with explicit bounds if necessary, though the previous call should have covered it.
    # Let's ensure we use the specific range for consistency.
    readings_df = data.readings(building_ids, start=start_ts, end=end_ts)

    all_mae_values = []
    
    for bid in building_ids:
        if bid not in readings_df.columns:
            continue
            
        series = readings_df[bid]
        
        # Check if we have enough valid data points for the history period
        valid_data = series.dropna()
        
        if len(valid_data) < history_days + horizon_hours:
            continue
            
        y = valid_data.values
        
        # Calculate number of possible windows
        n_windows = len(y) - (history_days + horizon_hours)
        
        if n_windows <= 0:
            continue
            
        total_error = 0.0
        count = 0
        
        for i in range(n_windows):
            train_end_idx = i + history_days
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + horizon_hours
            
            if test_end_idx > len(y):
                break
                
            # Training data: last `history_days` points
            train_data = y[train_end_idx - history_days : train_end_idx]
            
            # Test data: next `horizon_hours` points
            test_data = y[test_start_idx : test_end_idx]
            
            # Implement Seasonal Naive Forecast manually for efficiency and control
            # Determine seasonality period (hours per day)
            hours_per_day = 24
            
            # Calculate seasonal periods based on the training data length
            n_train_hours = len(train_data)
            num_seasons = n_train_hours // hours_per_day
            
            if num_seasons < 1:
                continue
                
            # Initialize predictions array
            predictions = np.zeros(len(test_data))
            
            for t in range(len(test_data)):
                # Find the corresponding index in the training data with the same time of day
                current_hour_of_day = (test_start_idx + t) % hours_per_day
                train_idx = train_end_idx - 1 - ((train_end_idx - 1 - current_hour_of_day) % n_train_hours)
                
                # More robust way: find the index in training data that has the same hour of day
                # We want y[t] = y[t - s] where s is the period to go back to the same time of day
                # The most recent occurrence of this hour in the training set
                target_hour_idx = (test_start_idx + t) % hours_per_day
                
                # Find the index in the training data that corresponds to this hour
                # We look for the last occurrence of this hour before test_start_idx
                # Since train_data ends at train_end_idx, we search backwards from there
                # The index in original series corresponding to target_hour_idx relative to test_start_idx
                
                # Calculate absolute index in original series
                abs_test_idx = test_start_idx + t
                # Find the matching hour in the training window
                # We want the largest index < train_end_idx such that (index % 24) == (abs_test_idx % 24)
                hour_of_day = abs_test_idx % hours_per_day
                
                # Search backwards from train_end_idx - 1 to find the matching hour
                search_start = max(0, train_end_idx - hours_per_day)
                matching_idx = None
                for idx in range(train_end_idx - 1, search_start - 1, -1):
                    if idx % hours_per_day == hour_of_day:
                        matching_idx = idx
                        break
                
                if matching_idx is not None:
                    predictions[t] = y[matching_idx]
                else:
                    # Fallback to the last available value if no exact match found (shouldn't happen with enough data)
                    predictions[t] = y[train_end_idx - 1]

            error = mean_absolute_error(test_data, predictions)
            total_error += error
            count += 1
        
        if count > 0:
            avg_mae = total_error / count
            all_mae_values.append(avg_mae)

    if not all_mae_values:
        raise ValueError("No valid prediction instances found for the specified criteria.")

    final_avg_mae = np.mean(all_mae_values)
    
    return {"answer": round(final_avg_mae, 6), "unit": "kWh"}
