"""EnergyX-Max program for template CS-054-T1.

For building {building_id}, how many hours in {year} have electricity consumption exceeding the mean by more than {sd_multiplier} standard deviations?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd

def answer(data, building_id, year, sd_multiplier):
    # Retrieve the specific building ID from the metadata to ensure it exists and get basic info if needed
    try:
        meta = data.metadata()
        if building_id not in meta.index:
            raise ValueError(f"Building {building_id} not found in metadata.")
        
        # Get start and end timestamps for the specified year
        start, end = data.year_range(year)
        
        # Read hourly consumption data for the specific building and year range
        readings = data.readings([building_id], start=start, end=end)
        
        # Check if there is any data
        if readings.empty:
            raise ValueError(f"No data found for building {building_id} in year {year}.")
        
        # Flatten the DataFrame to a 1D array of consumption values (kWh)
        # The 'readings' object returned by read_buildings usually has columns as building IDs.
        # We select the column corresponding to our building_id.
        if building_id not in readings.columns:
            raise ValueError(f"Building {building_id} not found in readings for year {year}.")
            
        consumption = readings[building_id].values
        
        # Filter out NaN values as per convention ("A reading is valid when it is not NaN")
        valid_consumption = consumption[~np.isnan(consumption)]
        
        if len(valid_consumption) == 0:
            raise ValueError(f"No valid readings found for building {building_id} in year {year}.")
            
        # Calculate mean and standard deviation using valid readings only
        mean_consumption = np.mean(valid_consumption)
        std_consumption = np.std(valid_consumption, ddof=0) # Population std dev usually implied unless specified sample
        
        # Calculate the threshold: mean + (sd_multiplier * std)
        threshold = mean_consumption + (sd_multiplier * std_consumption)
        
        # Count hours where consumption exceeds the threshold
        count_exceeding = np.sum(valid_consumption > threshold)
        
        return {"answer": float(count_exceeding), "unit": None}
        
    except Exception as e:
        raise ValueError(str(e))
