"""EnergyX-Max program for template CS-231-T2.

Using a seasonal naive forecast with {history_days} days of history, what is the average absolute percentage error for {building_type} buildings in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def answer(data, **slots):
    history_days = slots['history_days']
    building_type = slots['building_type']
    year = slots['year']
    
    start_ts, end_ts = data.year_range(year)
    buildings = data.buildings(building_type=building_type)
    qualifying = data.qualifying(buildings, year, min_share=0.9)
    
    if len(qualifying) == 0:
        raise ValueError("No qualifying buildings found")
        
    readings = data.readings(qualifying, start_ts, end_ts)
    readings.columns = [str(c) for c in readings.columns]
    
    valid_cols = [c for c in readings.columns if not readings[c].isna().all()]
    if len(valid_cols) == 0:
        raise ValueError("No valid readings found")
        
    readings = readings[valid_cols]
    
    forecasts = []
    actuals = []
    
    for col in readings.columns:
        series = readings[col]
        history_end = start_ts + pd.Timedelta(days=history_days)
        forecast_start = history_end
        forecast_end = end_ts
        
        if len(series) < history_days:
            continue
            
        history_series = series.iloc[-history_days:]
        forecast_values = []
        
        for i in range(len(forecast_series := pd.date_range(start=forecast_start, end=forecast_end, freq='h'))):
            idx = forecast_series[i]
            hist_idx = idx - pd.Timedelta(hours=24)
            if hist_idx in series.index:
                forecast_values.append(series[hist_idx])
            else:
                forecast_values.append(np.nan)
                
        pred_series = pd.Series(forecast_values, index=forecast_series)
        
        valid_mask = ~series.isna() & ~pred_series.isna()
        if valid_mask.sum() < 2:
            continue
            
        actuals.extend(series[valid_mask].values)
        forecasts.extend(pred_series[valid_mask].values)
            
    if len(actuals) == 0:
        raise ValueError("Insufficient data for error calculation")
        
    mape = mean_absolute_percentage_error(np.array(forecasts), np.array(actuals))
    return {"answer": round(mape, 6), "unit": "%"}
