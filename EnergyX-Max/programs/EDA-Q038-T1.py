"""EnergyX-Max program for template EDA-Q038-T1.

What is the {threshold_pct}th percentile of {building_id}'s hourly electricity demand during calendar year {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    threshold_pct = slots['threshold_pct']
    building_id = slots['building_id']
    year = slots['year']

    # Validate building_id exists in metadata
    if building_id not in data.metadata().index:
        raise ValueError(f"Building '{building_id}' not found in the dataset.")

    # Get start and end timestamps for the specified year
    start, end = data.year_range(year)

    # Read hourly electricity demand for the specific building and year
    readings_df = data.readings([building_id], start=start, end=end)

    if readings_df.empty:
        raise ValueError(f"No data found for building '{building_id}' in year {year}.")

    # Extract the column for the specific building
    demand_series = readings_df[building_id]

    # Filter out NaN values (invalid readings)
    valid_demand = demand_series.dropna()

    if len(valid_demand) == 0:
        raise ValueError(f"No valid hourly electricity consumption data found for building '{building_id}' in year {year}.")

    # Calculate the percentile
    # threshold_pct is expected to be a number (e.g., 20 for 20th percentile)
    percentile_value = np.percentile(valid_demand.values, float(threshold_pct))

    # Round to 6 decimals as per convention
    rounded_value = round(percentile_value, 6)

    return {"answer": rounded_value, "unit": "kWh"}
