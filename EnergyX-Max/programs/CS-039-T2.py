"""EnergyX-Max program for template CS-039-T2.

For all buildings in {city}, what is the average percentage of missing hourly readings per building?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    city = slots.get('city')
    
    # Get list of building IDs for the specified city
    try:
        building_ids = data.buildings(city=city)
    except Exception:
        raise ValueError(f"No buildings found in city: {city}")

    if not building_ids:
        raise ValueError(f"No buildings found in city: {city}")

    # Define the full year range to calculate missing readings for the whole year
    # Assuming we need to check against a standard year (e.g., 2023) or the most recent available.
    # Since the problem asks for "all buildings", and data might span multiple years, 
    # but usually these questions imply a specific context or the most complete dataset.
    # However, without a specified year in slots, we must infer or use a standard approach.
    # Looking at similar problems, often they imply checking against the full available history 
    # for each building or a specific reference year. 
    # Given the ambiguity of "all buildings" without a year constraint in the question text provided,
    # but the convention stating "A yearly measure ... uses buildings with at least 90% of the year's hours valid",
    # it implies we are looking at a specific year's performance. 
    # However, the question doesn't specify a year. Let's assume we calculate based on the 
    # entire available history for each building to get a robust average, OR check if there is a 
    # standard year implied. 
    # Re-reading: "For all buildings in {city}, what is the average percentage of missing hourly readings per building?"
    # It doesn't specify a year. The most logical interpretation for "missing hourly readings" without a date range 
    # is to consider the entire dataset available for those buildings.
    
    # Let's fetch readings for all these buildings. We need start and end.
    # Since no specific date range is given, we will use the full extent of data available in the 'readings' method.
    # The 'readings' method takes start=None, end=None which implies full history.
    
    try:
        df_readings = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to read data for buildings in {city}: {str(e)}")

    if df_readings.empty:
        raise ValueError("No readings found for the specified buildings.")

    # Calculate total hours available in the dataframe (should be consistent across columns)
    # The DatetimeIndex represents the full time range covered by the data source.
    total_hours = len(df_readings.index)
    
    if total_hours == 0:
        raise ValueError("No valid time range found in readings.")

    # Calculate missing hours per building
    missing_counts = df_readings.isna().sum()
    
    # Filter out buildings that have no data at all (all NaN), though the problem says "valid buildings".
    # Convention: "A yearly measure ... uses buildings with at least 90% of the year's hours valid".
    # Does this apply here? The question asks for "average percentage of missing hourly readings per building".
    # If a building has 0% data, is it included? 
    # Usually, in such contexts, we average over buildings that have *some* data.
    # However, the convention note about 90% might imply we should filter buildings with <90% validity if we are doing a "yearly measure".
    # But here we don't have a specific year. Let's assume we calculate for all buildings that have at least one reading.
    # Or, strictly following the convention: "A yearly measure ... uses buildings with at least 90% of the year's hours valid".
    # If we treat the entire history as "the year" (or the period of interest), we might filter.
    # But typically, "missing readings per building" implies looking at the data that exists.
    # Let's stick to calculating for all buildings present in the dataframe, excluding those with 0 total hours if any.
    # Actually, if a building is in `building_ids`, it should have some data or be excluded by the `readings` call? 
    # The `readings` method returns NaN where missing. If a building has no data at all, its column might be empty or all NaN.
    # Let's assume we include buildings that have at least one valid reading to avoid division by zero issues in percentage calc.
    
    valid_buildings = missing_counts[missing_counts < total_hours]
    
    if len(valid_buildings) == 0:
        raise ValueError("No buildings with valid readings found in the specified city.")

    # Calculate percentage of missing readings for each building
    percentages_missing = (valid_buildings / total_hours) * 100
    
    # Compute the average
    avg_percentage = percentages_missing.mean()

    return {"answer": round(avg_percentage, 6), "unit": "%"}
