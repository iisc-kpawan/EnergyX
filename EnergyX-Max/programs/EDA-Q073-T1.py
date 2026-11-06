"""EnergyX-Max program for template EDA-Q073-T1.

For building {building_id}, does electricity consumption show a heating sensitivity to outdoor air temperature, and if so, what is its heating change-point?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import t

def answer(data, **slots):
    building_id = slots['building_id']
    
    # Retrieve readings for the specific building
    try:
        readings_df = data.readings([building_id])
    except Exception:
        raise ValueError(f"Building {building_id} not found in data.")

    if building_id not in readings_df.columns:
        raise ValueError(f"Building {building_id} not found in data.")

    # Extract the specific column for this building
    col_name = building_id
    consumption_series = readings_df[col_name]
    
    # Get the weather data for the same period to align with consumption
    start_time = consumption_series.index.min()
    end_time = consumption_series.index.max() + pd.Timedelta(hours=1)
    
    try:
        meta = data.metadata()
        site_id = meta.loc[building_id, 'site_id']
        weather_df = data.weather(site_id=site_id, start=start_time, end=end_time)
    except Exception:
        raise ValueError(f"Could not retrieve weather data for building {building_id}.")

    # Align consumption and weather timestamps
    common_index = consumption_series.index.intersection(weather_df.index)
    
    if len(common_index) < 24: 
        raise ValueError(f"Not enough overlapping hourly data for building {building_id}.")

    consumption_aligned = consumption_series.reindex(common_index)
    temp_aligned = weather_df['air_temperature'].reindex(common_index)

    # Filter out NaNs in both series
    valid_mask = ~consumption_aligned.isna() & ~temp_aligned.isna()
    consumption_valid = consumption_aligned[valid_mask]
    temp_valid = temp_aligned[valid_mask]

    if len(consumption_valid) < 10:
        raise ValueError(f"Insufficient valid data points for building {building_id}.")

    # Define heating season logic: typically when outdoor temperature is below a threshold (e.g., 15°C or 20°C).
    t_threshold = 15.0
    
    # Identify heating hours
    heating_mask = temp_valid <= t_threshold
    
    if heating_mask.sum() < 24:
        raise ValueError(f"No significant heating season data found for building {building_id} (threshold {t_threshold}°C).")

    consumption_heating = consumption_valid[heating_mask]
    temp_heating = temp_valid[heating_mask]

    # Fit a linear regression on the heating period: Consumption = alpha + beta * Temp
    X_heating = temp_heating.values.reshape(-1, 1)
    y_heating = consumption_heating.values

    model = LinearRegression()
    model.fit(X_heating, y_heating)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X_heating, y_heating)

    # Determine if sensitivity is significant. 
    n = len(consumption_heating)
    y_mean = np.mean(y_heating)
    y_std = np.std(y_heating)
    x_std = np.std(temp_heating)
    
    # Standard error of slope: SE(b) = s / (sqrt(n) * x_std) where s is std of residuals
    residuals = y_heating - model.predict(X_heating)
    s = np.std(residuals, ddof=1)
    
    if n > 2:
        se_slope = s / (np.sqrt(n) * x_std) if x_std != 0 else float('inf')
        t_stat = slope / se_slope if se_slope != 0 else float('inf')
        df = n - 2
        p_val = 2 * (1 - t.cdf(abs(t_stat), df))
    else:
        p_val = 1.0

    is_significant = False
    if p_val < 0.05 and abs(slope) > 0.01: 
        is_significant = True
        
    # If significant, the change point is the threshold temperature where heating demand starts to increase linearly with temp drop.
    # In this simplified model (fixed threshold), the change point is the threshold itself.
    # However, if we want to estimate the optimal breakpoint from data:
    
    if is_significant:
        # Estimate optimal breakpoint by fitting piecewise linear regression
        all_temps = temp_valid.values
        all_cons = consumption_valid.values
        
        min_t = np.min(all_temps)
        max_t = np.max(all_temps)
        
        best_bp = None
        best_error = float('inf')
        
        # Grid search for breakpoint with step 0.5C
        steps = np.arange(min_t, max_t, 0.5)
        
        for bp in steps:
            mask_left = all_temps <= bp
            mask_right = all_temps > bp
            
            if mask_left.sum() < 10 or mask_right.sum() < 10:
                continue
                
            X_left = all_temps[mask_left].reshape(-1, 1)
            y_left = all_cons[mask_left]
            
            X_right = all_temps[mask_right].reshape(-1, 1)
            y_right = all_cons[mask_right]
            
            model_left = LinearRegression().fit(X_left, y_left)
            model_right = LinearRegression().fit(X_right, y_right)
            
            pred_left = model_left.predict(X_left)
            pred_right = model_right.predict(X_right)
            
            total_error = np.sum((y_left - pred_left)**2) + np.sum((y_right - pred_right)**2)
            
            if total_error < best_error:
                best_error = total_error
                best_bp = bp
        
        change_point = best_bp
    else:
        # If not significant, we still return the threshold as the theoretical change point but note no sensitivity.
        # However, the question asks "does it show... and if so". If not significant, we might say no.
        # But to provide a record, let's assume the threshold is the change point regardless of significance for reporting purposes,
        # or set change_point to None/NaN if truly no sensitivity.
        # Given the instruction "if so", we should probably indicate 'No' in the record.
        # Let's set change_point to the threshold but mark sensitivity as False.
        change_point = t_threshold

    result = {
        "sensitive": is_significant,
        "change_point": round(change_point, 2) if is_significant else None,
        "slope": round(slope, 6),
        "r_squared": round(r_squared, 6)
    }
    
    return {"answer": result, "unit": None}
