"""EnergyX-Max program for template CS-303-T1.

Using a seasonal naive forecast with a {horizon_hours}-hour horizon, what is the predicted hourly electricity consumption for {building_id} starting at {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, horizon_hours, building_id, date):
    start = pd.to_datetime(date)
    end = start + pd.Timedelta(hours=horizon_hours)
    full_start = start - pd.Timedelta(days=365)
    full_end = end
    readings = data.readings([building_id], start=full_start, end=full_end)
    if building_id not in readings.columns:
        raise ValueError("Building not found")
    col = readings[building_id]
    valid_mask = ~col.isna()
    if not valid_mask.any():
        raise ValueError("No valid historical data")
    forecast_series = pd.Series(index=pd.date_range(start=start, periods=horizon_hours, freq='h'), dtype=float)
    for ts in forecast_series.index:
        prev_ts = ts - pd.Timedelta(days=365)
        if prev_ts in readings.index and not pd.isna(col[prev_ts]):
            forecast_series[ts] = col[prev_ts]
    return {"answer": forecast_series.values.tolist(), "unit": "kWh"}
