"""EnergyX-Max program for template EDA-Q042-T2.

What is the average hourly load difference between weekdays and weekends for {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_id = slots['building_id']
    
    # Retrieve hourly readings for the specific building
    # The read function returns a wide DataFrame where columns are building IDs and rows are timestamps
    try:
        df = data.readings([building_id])
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if building_id not in df.columns:
        raise ValueError(f"Building ID '{building_id}' not found in the dataset.")

    # Extract the load series for this building
    load_series = df[building_id]

    # Ensure we have a valid DatetimeIndex to determine days of the week
    if not isinstance(load_series.index, pd.DatetimeIndex):
        raise ValueError("Readings index is not a DatetimeIndex.")

    # Create a copy to avoid SettingWithCopyWarning and ensure we work with the correct series
    load_df = load_series.to_frame(name='load')
    
    # Determine the day of the week (0=Monday, ..., 6=Sunday)
    load_df['day_of_week'] = load_df.index.dayofweek
    
    # Define weekdays as Mon-Fri (0-4) and weekends as Sat-Sun (5-6)
    is_weekday = (load_df['day_of_week'].isin([0, 1, 2, 3, 4]))
    is_weekend = (load_df['day_of_week'].isin([5, 6]))
    
    # Filter data for weekdays and weekends
    weekday_loads = load_df.loc[is_weekday, 'load']
    weekend_loads = load_df.loc[is_weekend, 'load']

    # Check if there is enough data to compute the average
    if len(weekday_loads) == 0 or len(weekend_loads) == 0:
        raise ValueError(f"No valid weekday or weekend data found for building {building_id}.")

    # Calculate averages using only non-NaN values (though filtering by day usually handles this, 
    # explicit dropna is safer if the underlying data has NaNs that align with specific days)
    avg_weekday_load = weekday_loads.dropna().mean()
    avg_weekend_load = weekend_loads.dropna().mean()

    # Calculate the difference (Weekday - Weekend) as per standard interpretation of "difference between A and B"
    # If the question implies magnitude, we might take abs(), but usually "difference between X and Y" is X - Y.
    # Given the context of load profiles, it's often interesting to see how much higher weekdays are.
    difference = avg_weekday_load - avg_weekend_load

    return {"answer": round(float(difference), 6), "unit": "kWh"}
