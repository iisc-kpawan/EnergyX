"""EnergyX-Max program for template EDA-Q033-T2.

What is the highest peak-to-average hourly electricity consumption ratio among {building_type} buildings in {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, building_type, country):
    # Filter buildings by country and building type
    try:
        building_ids = data.buildings(country=country, building_type=building_type)
    except Exception:
        # Handle cases where the filter arguments might differ slightly or raise errors
        # Assuming standard API behavior where None is default for unspecified filters
        all_bldgs = data.buildings()
        # Filter manually based on metadata if direct filtering fails or isn't supported with multiple args
        meta = data.metadata()
        mask = (meta['country'] == country) & (meta['building_type'] == building_type)
        building_ids = meta[mask].index.tolist()

    if not building_ids:
        raise ValueError(f"No {building_type} buildings found in {country}.")

    # Read all hourly data for these buildings
    # The readings function returns a wide DataFrame with building IDs as columns
    try:
        df_readings = data.readings(building_ids)
    except Exception:
        raise ValueError("Failed to retrieve electricity consumption data.")

    if df_readings.empty:
        raise ValueError("No valid hourly consumption data found for the specified buildings.")

    # Calculate peak (max) and average (mean) for each building
    # We use .max() and .mean() which ignore NaNs by default, but we need to ensure 
    # we are calculating over the available hours. The problem implies using valid readings.
    # However, "average" usually implies mean of all recorded hours or mean of valid hours.
    # Given "A reading is valid when it is not NaN", and standard practice for such metrics:
    # We calculate max and mean of the non-NaN values for each building.
    
    peak_avg_ratios = []
    
    for col in df_readings.columns:
        series = df_readings[col]
        
        # Get valid (non-NaN) readings
        valid_series = series.dropna()
        
        if len(valid_series) == 0:
            continue
            
        peak = valid_series.max()
        average = valid_series.mean()
        
        if average == 0:
            # Avoid division by zero. If average is 0, ratio is undefined or 0 depending on context.
            # Usually implies no consumption, so ratio is 0 or NaN. Let's treat as 0 for safety 
            # or skip. Given "highest peak-to-average", if avg is 0 and peak > 0, it's infinite.
            # But physically, if average is 0, there is no consumption. Let's assume valid data has non-zero avg.
            # If peak is also 0, ratio is 0.
            if peak == 0:
                ratio = 0.0
            else:
                # This case (peak > 0, avg = 0) implies a single hour of consumption or very sparse data.
                # Mathematically undefined, but practically we might skip or set to a large number.
                # However, for "highest ratio", if this happens, it's an outlier. 
                # Let's assume well-behaved data where avg > 0 for buildings with consumption.
                # If we must handle it: let's say ratio is 0 to avoid infinity in a list of numbers.
                ratio = 0.0 
        else:
            ratio = peak / average
        
        peak_avg_ratios.append(ratio)

    if not peak_avg_ratios:
        raise ValueError("No valid consumption data found for calculation.")

    ratios_array = np.array(peak_avg_ratios)
    
    # Find the highest ratio
    max_ratio = float(np.max(ratios_array))
    
    return {"answer": round(max_ratio, 6), "unit": None}
