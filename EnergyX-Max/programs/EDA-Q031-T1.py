"""EnergyX-Max program for template EDA-Q031-T1.

What are the mean and median hourly electricity demand for all {building_type} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    import numpy as np
    import pandas as pd
    
    building_type = slots.get('building_type')
    
    # Get list of building IDs for the specified building type
    building_ids = data.buildings(building_type=building_type)
    
    if not building_ids:
        raise ValueError(f"No buildings found with building type '{building_type}'.")
    
    # Read hourly readings for all selected buildings
    # We assume we want all available data, so no start/end constraints unless specified by context (none here)
    df = data.readings(building_ids)
    
    if df.empty:
        raise ValueError("No hourly readings found for the specified buildings.")
    
    # Calculate mean and median for each building
    means = df.mean(axis=0)
    medians = df.median(axis=0)
    
    # Create a DataFrame to hold the results
    result_df = pd.DataFrame({
        'building_id': means.index,
        'mean_kwh': means.values,
        'median_kwh': medians.values
    })
    
    # Sort by building_id for consistent ordering
    result_df = result_df.sort_values('building_id').reset_index(drop=True)
    
    # Convert to a dictionary of records as expected by the "record" answer type
    records = result_df.to_dict(orient='records')
    
    return {"answer": records, "unit": "kWh"}
