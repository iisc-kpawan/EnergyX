"""EnergyX-Max program for template ADV-04-T2.

How does the average MAPE of a seasonal naive forecast compare to an ARIMA model for {building_type} buildings at the {site_id} site during {month}, using a 24-hour horizon?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, site_id, month) -> dict:
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_percentage_error
    
    # 1. Identify buildings
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for site_id={site_id} and building_type={building_type}")
    
    if not building_ids:
        raise ValueError("No qualifying buildings found.")

    # 2. Define time range
    start_ts, end_ts = data.month_range(month)
    
    # 3. Get readings
    readings_df = data.readings(building_ids, start=start_ts, end=end_ts)
    
    if readings_df.empty:
        raise ValueError("No energy readings found for the specified period.")

    horizon = 24
    
    mape_seasonal_naive_list = []
    mape_arima_list = []

    # Iterate over buildings
    for bid in building_ids:
        col_name = f"{bid}"
        if col_name not in readings_df.columns:
            continue
            
        series = readings_df[col_name]
        
        # Ensure we have enough data points
        if len(series) < horizon + 24: 
            continue
            
        # --- Seasonal Naive Forecast ---
        # Strategy: For each valid start time t, predict the next 'horizon' hours (t+1 to t+horizon)
        # using values from t-23 to t.
        
        mape_sum = 0.0
        count = 0
        
        for t in range(len(series) - horizon):
            start_hist_idx = t - 23
            end_hist_idx = t + 1
            
            if start_hist_idx < 0:
                continue
                
            hist_values = series.iloc[start_hist_idx:end_hist_idx].values
            actual_start = t + 1
            actual_end = t + horizon + 1
            
            # Check bounds for actuals
            if actual_end > len(series):
                break
                
            actual_values = series.iloc[actual_start:actual_end].values
            
            # Check for NaN in history used for prediction or actuals
            if np.any(np.isnan(hist_values)) or np.any(np.isnan(actual_values)):
                continue
                
            # Avoid division by zero
            if np.any(actual_values == 0):
                continue
                
            pred_vector = hist_values
            actual_valid = actual_values
            
            mape = mean_absolute_percentage_error(actual_valid, pred_vector)
            mape_sum += mape
            count += 1
        
        if count > 0:
            avg_mape_seasonal_naive = mape_sum / count
            mape_seasonal_naive_list.append(avg_mape_seasonal_naive)

    # --- ARIMA Forecast ---
    # Strategy: Fit ARIMA on the available data up to time t, then predict next 'horizon' hours.
    # To ensure stability and avoid overfitting on very short series, we might need a minimum history.
    # However, for comparison within the same window, we will fit on all available data up to t-1
    # and predict t...t+horizon-1? 
    # Standard practice: Fit on history, predict forward.
    # We will iterate similarly: at time t (where t is the last index of history), fit ARIMA on series[:t],
    # then predict horizon steps.
    
    for bid in building_ids:
        col_name = f"{bid}"
        if col_name not in readings_df.columns:
            continue
            
        series = readings_df[col_name]
        
        if len(series) < horizon + 24: 
            continue
            
        mape_sum = 0.0
        count = 0
        
        # We need at least some history to fit ARIMA. Let's start fitting from index 24 onwards.
        for t in range(24, len(series) - horizon):
            # Fit ARIMA on data up to t (exclusive of t? or inclusive?)
            # Usually, we fit on series[:t] and predict starting at t.
            # But let's assume we fit on series[:t+1] and predict t+1...t+horizon.
            # Let's stick to: Fit on series[:t], predict t...t+horizon-1? 
            # The question asks for a 24-hour horizon forecast error.
            # Let's assume we fit on series[:t] and predict the next 'horizon' steps starting at t.
            
            history = series.iloc[:t].dropna()
            if len(history) < 10: # Minimum data points for ARIMA
                continue
                
            try:
                model = ARIMA(history, order=(1,1,1))
                fitted_model = model.fit()
                
                # Predict next 'horizon' steps
                forecast = fitted_model.forecast(steps=horizon)
                
                # Actual values corresponding to the forecast
                actual_start = t
                actual_end = t + horizon
                actual_values = series.iloc[actual_start:actual_end].values
                
                if np.any(np.isnan(forecast)) or np.any(np.isnan(actual_values)):
                    continue
                    
                if np.any(actual_values == 0):
                    continue
                    
                mape = mean_absolute_percentage_error(actual_values, forecast)
                mape_sum += mape
                count += 1
            except Exception:
                # If ARIMA fails (e.g., convergence issues), skip this time step
                continue
        
        if count > 0:
            avg_mape_arima = mape_sum / count
            mape_arima_list.append(avg_mape_arima)

    # Compile results
    if not mape_seasonal_naive_list or not mape_arima_list:
        raise ValueError("Could not compute MAPE for the specified models and buildings.")
        
    avg_mape_seasonal_naive = np.mean(mape_seasonal_naive_list)
    avg_mape_arima = np.mean(mape_arima_list)
    
    result = {
        "Seasonal Naive": round(avg_mape_seasonal_naive, 6),
        "ARIMA": round(avg_mape_arima, 6)
    }
    
    return {"answer": result, "unit": None}
