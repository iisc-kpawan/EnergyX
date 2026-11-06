"""EnergyX-Max program for template CS-056-T2.

Using a seasonal naive forecast with {history_days} days of history, identify all buildings in {country} where the absolute residual exceeds {threshold_pct}% for more than 10 consecutive hours during {year}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    history_days = slots['history_days']
    country = slots['country']
    threshold_pct = slots['threshold_pct']
    year = slots['year']
    
    # 1. Identify buildings in the specified country
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in country '{country}'.")

    if not building_ids:
        raise ValueError(f"No buildings found in country '{country}'.")

    # 2. Define the time range for the specific year
    start_ts, end_ts = data.year_range(year)
    
    # 3. Filter buildings that qualify (>= 90% valid hours usually required for yearly stats unless specified otherwise)
    qualifying_bids = data.qualifying(building_ids, year, min_share=0.9)
    
    if not qualifying_bids:
        raise ValueError(f"No qualifying buildings found in '{country}' for year '{year}'.")

    # 4. Prepare readings for the full year range
    try:
        df_readings = data.readings(building_ids=qualifying_bids, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    # Ensure the index is DatetimeIndex and sorted
    if not isinstance(df_readings.index, pd.DatetimeIndex):
        df_readings = df_readings.sort_index()
        df_readings.index = pd.to_datetime(df_readings.index)
    
    # Create a mask for the specific year hours within the provided range
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59.999999") 
    mask = (df_readings.index >= year_start) & (df_readings.index < year_end)
    df_yearly = df_readings.loc[mask]
    
    if df_yearly.empty:
        raise ValueError(f"No data available for year '{year}'.")

    threshold_val = threshold_pct / 100.0
    identified_buildings = []

    for bid in qualifying_bids:
        if bid not in df_yearly.columns:
            continue
            
        series = df_yearly[bid]
        
        # Shift the series by 24 hours to get the forecast values aligned with current time
        s_shifted = series.shift(24)
        
        # Calculate absolute residuals
        abs_residuals = (series - s_shifted).abs()
        
        # Check for consecutive hours exceeding threshold
        exceed_mask = abs_residuals > threshold_val
        
        if not exceed_mask.any():
            continue
            
        # Find consecutive runs of True values
        # Create a group identifier where value changes from False to True or True to False
        is_true = exceed_mask.values.astype(int)
        
        # Use numpy to find run lengths
        # Pad with 0 at start and end to handle boundaries
        padded = np.concatenate(([0], is_true, [0]))
        diff = np.diff(padded)
        change_indices = np.where(diff != 0)[0]
        
        if len(change_indices) < 2:
            continue
            
        # Extract run lengths
        runs = []
        start_idx = change_indices[0] + 1
        for i in range(1, len(change_indices)):
            end_idx = change_indices[i]
            length = end_idx - start_idx
            if is_true[start_idx:end_idx].sum() == length:
                runs.append(length)
            start_idx = change_indices[i] + 1
        
        # Check if any run exceeds 10 hours
        if any(run > 10 for run in runs):
            identified_buildings.append(bid)

    return {"answer": sorted(identified_buildings), "unit": None}
