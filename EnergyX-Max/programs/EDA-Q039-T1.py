"""EnergyX-Max program for template EDA-Q039-T1.

What proportion of {building_id}'s annual electricity consumption occurred during its top {top_n}% highest-demand hours?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd

def answer(data, building_id, top_n):
    # Validate inputs
    if not isinstance(building_id, str) or not isinstance(top_n, (int, float)):
        raise ValueError("Invalid input types for building_id and top_n.")
    
    if top_n <= 0 or top_n > 100:
        raise ValueError("top_n must be between 0 and 100.")

    # Retrieve annual consumption data for the specified building.
    # The problem states to include all hours in 2016-2017.
    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2018-01-01') # Exclusive end for range
    
    try:
        readings = data.readings([building_id], start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")

    if readings.empty or building_id not in readings.columns:
        raise ValueError(f"No data found for building {building_id} in the specified period.")

    # Extract the consumption series for the specific building
    consumption_series = readings[building_id]

    # Calculate total annual consumption (sum of valid readings)
    total_annual_consumption = consumption_series.sum()

    if pd.isna(total_annual_consumption):
        raise ValueError(f"No valid electricity consumption data found for building {building_id} in 2016-2017.")

    # Calculate the threshold value for the top_n percentage
    threshold_value = total_annual_consumption * (top_n / 100.0)

    # Identify hours where consumption exceeds the threshold
    high_demand_hours_mask = consumption_series > threshold_value
    
    # Sum consumption during these high-demand hours
    consumption_during_top_hours = consumption_series[high_demand_hours_mask].sum()

    if pd.isna(consumption_during_top_hours):
        return {"answer": 0.0, "unit": None}

    # Calculate the proportion
    proportion = consumption_during_top_hours / total_annual_consumption

    # Round to 6 decimals as per convention
    proportion_rounded = round(proportion, 6)

    return {"answer": proportion_rounded, "unit": None}
