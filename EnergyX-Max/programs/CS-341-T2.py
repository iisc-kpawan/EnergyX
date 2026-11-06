"""EnergyX-Max program for template CS-341-T2.

Compare the anomaly detection performance of two methods across all {building_type} buildings in {country} for the year {year}, reporting the count of flagged anomalies per method.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_type = slots['building_type']
    country = slots['country']
    year = int(slots['year'])
    
    bldgs = data.buildings(building_type=building_type, country=country)
    if not bldgs:
        raise ValueError(f"No {building_type} buildings found in {country}")
    
    start, end = data.year_range(year)
    readings_df = data.readings(bldgs, start=start, end=end)
    
    qualifying = data.qualifying(bldgs, year=year, min_share=0.9)
    if not set(qualifying).issubset(set(readings_df.columns)):
        qualifying = [b for b in readings_df.columns if readings_df[b].notna().sum() / 8760 >= 0.9]
    
    if not qualifying:
        raise ValueError("No qualifying buildings with sufficient data")
    
    readings_subset = readings_df[qualifying]
    
    def detect_anomalies(series):
        mean_val = series.mean()
        std_val = series.std()
        threshold = 3 * std_val
        anomalies = (series > mean_val + threshold) | (series < mean_val - threshold)
        return anomalies.sum()
    
    method1_counts = readings_subset.apply(detect_anomalies, axis=0).to_dict()
    
    def detect_anomalies_weather(series, weather_series):
        corr = series.corr(weather_series)
        if pd.isna(corr):
            return 0
        threshold = 2 * abs(corr)
        mean_val = series.mean()
        anomalies = (series > mean_val + threshold) | (series < mean_val - threshold)
        return anomalies.sum()

    site_id = data.metadata().loc[bldgs[0], 'site_id']
    weather_df = data.weather(site_id, start=start, end=end)
    
    if weather_df.empty:
        method2_counts = {b: 0 for b in qualifying}
    else:
        method2_counts = readings_subset.apply(lambda s: detect_anomalies_weather(s, weather_df['air_temperature']), axis=0).to_dict()

    return {"answer": {"Method 1 (Statistical)": method1_counts, "Method 2 (Weather Correlation)": method2_counts}, "unit": None}
