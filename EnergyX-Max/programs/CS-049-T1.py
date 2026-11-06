"""EnergyX-Max program for template CS-049-T1.

For building {building_id}, how many hours in {year} have electricity consumption exceeding the mean by more than {sd_multiplier} standard deviations?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd

def answer(data, building_id, year, sd_multiplier):
    # Get the time range for the specified year
    start_ts, end_ts = data.year_range(year)
    
    # Read consumption data for the specific building and year
    try:
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
    
    # Extract the consumption series for the building (it should be a Series or DataFrame with one column)
    if isinstance(readings_df, pd.DataFrame):
        consumption = readings_df[building_id]
    else:
        consumption = readings_df
    
    # Filter out NaN values to calculate valid statistics
    valid_consumption = consumption.dropna()
    
    if len(valid_consumption) == 0:
        raise ValueError(f"No valid hourly consumption data found for building {building_id} in year {year}.")

    # Calculate mean and standard deviation based on valid readings only
    mean_consumption = valid_consumption.mean()
    std_consumption = valid_consumption.std()
    
    if np.isnan(mean_consumption) or np.isnan(std_consumption):
        raise ValueError(f"Mean or standard deviation is undefined for building {building_id} in year {year}.")

    # Calculate the threshold: mean + (sd_multiplier * std)
    threshold = mean_consumption + (sd_multiplier * std_consumption)
    
    # Count hours where consumption exceeds the threshold
    exceeding_hours = (valid_consumption > threshold).sum()
    
    return {"answer": float(exceeding_hours), "unit": None}
