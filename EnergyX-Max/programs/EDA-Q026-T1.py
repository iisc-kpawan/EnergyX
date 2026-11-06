"""EnergyX-Max program for template EDA-Q026-T1.

For building {building_id}, how many hourly consumption readings exceed the mean by more than {sd_multiplier} standard deviations between 2016 and 2017?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, sd_multiplier):
    import numpy as np
    
    # Retrieve readings for the specific building
    # The read function returns a wide DataFrame where columns are building IDs.
    # We need to extract the column corresponding to the given building_id.
    try:
        df_readings = data.readings(building_ids=[building_id], start=None, end=None)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for building {building_id}: {str(e)}")
    
    # Check if the building exists in the data
    if building_id not in df_readings.columns:
        raise ValueError(f"Building {building_id} not found in the dataset.")

    # Extract the time series for this building
    consumption = df_readings[building_id]
    
    # Filter data between 2016 and 2017 (inclusive)
    # The year_range method returns start and end timestamps. 
    # We assume 'end' is exclusive as per documentation.
    start_2016, end_2017 = data.year_range(2017)
    
    # Slice the dataframe to get only 2016 and 2017 data
    # Note: year_range(2017) likely returns start of 2017 to end of 2017 (exclusive).
    # We need from start of 2016 to end of 2017.
    # Let's get the range for 2016 and 2017 separately or construct the full range.
    # Actually, year_range(year) usually gives the slice for that specific year.
    # To get 2016-2017, we can get start of 2016 and end of 2017.
    
    # Let's adjust: get start of 2016 and end of 2017 (exclusive)
    start_2016 = data.year_range(2016)[0]
    end_2017 = data.year_range(2017)[1]
    
    # Filter the consumption series
    mask = (consumption.index >= start_2016) & (consumption.index < end_2017)
    consumption_filtered = consumption[mask]
    
    if len(consumption_filtered) == 0:
        raise ValueError("No data available for the specified building between 2016 and 2017.")

    # Calculate mean and standard deviation
    mean_val = consumption_filtered.mean()
    std_val = consumption_filtered.std(ddof=1) # Sample standard deviation is standard for such stats
    
    if np.isnan(mean_val) or np.isnan(std_val):
        raise ValueError("Invalid data (NaN mean or std) found for the specified period.")

    # Calculate threshold: mean + sd_multiplier * std
    threshold = mean_val + sd_multiplier * std_val
    
    # Count readings exceeding the threshold
    count_exceeding = (consumption_filtered > threshold).sum()
    
    return {"answer": float(count_exceeding), "unit": None}
