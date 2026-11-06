"""EnergyX-Max program for template EDA-Q077-T1.

What percentile does {building_id} fall into for total consumption among all {building_type} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import pandas as pd
    import numpy as np
    
    building_id = slots['building_id']
    building_type = slots['building_type']

    # 1. Identify the population of buildings for comparison
    # Filter by building_type from metadata
    try:
        pop_building_ids = data.buildings(building_type=building_type)
    except Exception as e:
        raise ValueError(f"No buildings found with type '{building_type}'.")

    if not pop_building_ids:
        raise ValueError(f"No buildings found with type '{building_type}' to compare against.")

    # 2. Retrieve consumption data for the population and the target building
    # We need the full available history. The read method handles this when start/end are None.
    try:
        pop_readings = data.readings(pop_building_ids)
    except Exception as e:
        raise ValueError(f"Error retrieving readings for population buildings: {str(e)}")

    if building_id not in pop_readings.columns:
        # The target building might be of a different type or simply not in the dataset's metered list
        raise ValueError(f"Building '{building_id}' was not found in the dataset or does not match the specified type.")

    # 3. Calculate total consumption for each building in the population
    # Sum over all hours (NaNs are ignored by sum)
    totals = pop_readings.sum(axis=0)
    
    target_total = totals[building_id]
    
    if pd.isna(target_total):
        raise ValueError(f"No valid consumption data found for building '{building_id}'.")

    # 4. Calculate percentiles for the population
    # Create a series of totals for the population
    pop_totals = pd.Series(totals)
    
    # Count how many are strictly less than target, and how many are equal to target
    count_less = (pop_totals < target_total).sum()
    count_equal = (pop_totals == target_total).sum()
    n = len(pop_totals)
    
    # PERCENTRANK.INC formula: (n_less + 0.5 * n_equal) / N * 100
    percentile_val = (count_less + 0.5 * count_equal) / n * 100
    
    # Round to 6 decimals as per convention
    result = round(percentile_val, 6)

    return {"answer": result, "unit": None}
