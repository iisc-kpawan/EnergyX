"""EnergyX-Max program for template CS-038-T1.

Which buildings have more than {threshold_pct}% missing hourly electricity consumption readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    threshold_pct = slots.get('threshold_pct')
    
    # Get all building IDs from metadata (metered buildings)
    try:
        meta_df = data.metadata()
        if meta_df.empty:
            raise ValueError("No metered buildings found in the dataset.")
        
        building_ids = meta_df.index.tolist()
    except Exception as e:
        raise ValueError(f"Error accessing metadata: {str(e)}")

    # If no threshold is provided or it's invalid, we might need to handle it.
    # Assuming threshold_pct is a number between 0 and 100.
    if not isinstance(threshold_pct, (int, float)):
        raise ValueError("threshold_pct must be a numeric value.")
    
    if threshold_pct < 0 or threshold_pct > 100:
        raise ValueError("threshold_pct must be between 0 and 100.")

    # We need to calculate the percentage of missing readings for each building.
    # Since we don't have a specific year/month/date range in slots, 
    # we will assume the question implies "overall" or "for the available data".
    # However, without a time range, 'readings' might return everything available.
    # Let's fetch readings for all buildings without time constraints to get the full history.
    
    try:
        # Fetch readings for all buildings. 
        # The function signature suggests start/end are optional.
        # If we pass empty lists or None, it should return all data.
        # Based on typical API patterns, passing no args or specific logic is needed.
        # Let's assume calling data.buildings() gives IDs, then we read for them.
        # To get the full dataset for missing calculation, we call readings with no time filter.
        
        # Re-reading the docstring: data.readings(building_ids, start=None, end=None)
        # If we pass building_ids and None for start/end, it should return all.
        df_readings = data.readings(building_ids, start=None, end=None)
    except Exception as e:
        raise ValueError(f"Error fetching readings: {str(e)}")

    if df_readings.empty:
        raise ValueError("No reading data found.")

    # Calculate missing percentage for each building column
    # Missing count = total hours - valid (non-NaN) hours
    # Total hours is len(df_readings)
    
    total_hours = len(df_readings)
    
    if total_hours == 0:
        raise ValueError("No hourly data points found in the dataset.")

    missing_counts = df_readings.isna().sum()
    
    # Calculate percentage of missing readings
    missing_pct = (missing_counts / total_hours) * 100
    
    # Filter buildings where missing percentage > threshold_pct
    # Note: The question asks "more than {threshold_pct}%", so strictly greater than.
    qualifying_buildings = missing_pct[missing_pct > threshold_pct].index.tolist()

    # Sort the list of building IDs as per convention for sets/lists in answers usually implies sorted order
    qualifying_buildings.sort()

    if not qualifying_buildings:
        # If no buildings meet the criteria, return an empty list. 
        # The instructions say "If the population is empty ... raise ValueError".
        # However, returning an empty set/list is a valid answer for "Which buildings...".
        # But let's check if "population" refers to the input slots or the result.
        # Usually, if no buildings match, returning [] is correct unless it implies error on empty dataset.
        # Given "If the population is empty", this likely refers to the dataset being empty, not the result set.
        return {"answer": [], "unit": None}

    return {"answer": qualifying_buildings, "unit": None}
