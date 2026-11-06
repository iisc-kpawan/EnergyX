"""EnergyX-Max program for template CS-058-T1.

Which {building_type} buildings have less than {threshold_pct}% missing electricity observations?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    threshold_pct = float(slots['threshold_pct'])
    
    # Get list of buildings for the specific type
    try:
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        # Fallback if direct filtering fails, try iterating all and checking type
        all_buildings = data.buildings()
        building_ids = [bid for bid in all_buildings if data.metadata().loc[bid, 'building_type'] == building_type]
        
    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}'.")

    # Define the year range to check missing observations. 
    # Since no specific year is provided in slots, we assume the question implies 
    # checking against a standard reference (e.g., 2023) or all available years.
    # However, the convention states "A yearly measure ... uses buildings with at least 90% of the year's hours valid".
    # Without a specific year slot, we must infer. Usually, such questions refer to the most recent full year 
    # or a specific year if provided. Given the lack of 'year' in slots, let's check the most recent full year available 
    # or raise an error if ambiguous. 
    # Let's try to get the range for 2023 first. If that fails or is empty, we might need to adjust.
    # Actually, looking at the schema, 'year' is a slot. It is missing here. 
    # Re-reading the prompt: "Slots: ['building_type', 'threshold_pct']". Year is NOT in slots.
    # This implies the check should be done over the entire dataset's history or a default year.
    # Given the ambiguity, and the instruction "A yearly measure ... uses buildings with at least 90% of the year's hours valid",
    # it is highly likely the question expects us to use a specific year if available, OR check all years?
    # But "missing electricity observations" usually implies a rate over a period.
    # Let's assume the question refers to the most recent full year for which data exists, 
    # or we must ask for clarification. However, as an agent, I must provide code.
    # Let's try to determine the latest full year from metadata if possible, or default to 2023 if it exists.
    
    # Strategy: Try to find the latest full year in the data.
    # We can sample one building to check its date range.
    if building_ids:
        # Get readings for one building to determine date range
        try:
            sample_readings = data.readings(building_ids=[building_ids[0]])
            if not sample_readings.empty:
                start_date = sample_readings.index.min()
                end_date = sample_readings.index.max()
                
                # Calculate the latest full year
                from datetime import datetime, timedelta
                latest_year_end = end_date.replace(month=12, day=31)
                latest_year_start = latest_year_end.replace(year=latest_year_end.year)
                
                # Check if there is data for this year
                year_range = (latest_year_start, latest_year_end)
            else:
                raise ValueError("No readings found.")
        except Exception:
             raise ValueError("Could not determine date range for missing observation check.")
    else:
        raise ValueError("No buildings found.")

    # Define start and end for the selected year
    try:
        start_ts, end_ts = data.year_range(latest_year_start.year)
    except Exception:
        # If year_range fails or we can't determine a year, maybe check all years? 
        # But "less than X% missing" is a yearly metric. Let's assume the latest full year.
        # If year_range doesn't support dynamic year construction easily, we use the timestamps derived above.
        start_ts = latest_year_start
        end_ts = latest_year_end

    # Filter buildings that have at least 90% valid hours for this year (Convention)
    # The convention says: "A yearly measure ... uses buildings with at least 90% of the year's hours valid"
    # So we first filter by qualifying, then check the missing percentage.
    
    qualifying_buildings = data.qualifying(building_ids, latest_year_start.year, min_share=0.9)
    
    # If no buildings qualify, return empty set? Or raise error? 
    # Convention implies we only consider qualifying buildings for yearly measures.
    if not qualifying_buildings:
        return {"answer": [], "unit": None}

    # Readings for the qualifying buildings for the specific year
    try:
        df_readings = data.readings(building_ids=qualifying_buildings, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Error reading data for year {latest_year_start.year}: {str(e)}")

    # Calculate missing percentage for each building
    total_hours = len(df_readings.index)
    if total_hours == 0:
        raise ValueError("No valid hours found in the specified period.")

    missing_counts = df_readings.isna().sum(axis=1)
    missing_pcts = (missing_counts / total_hours) * 100

    # Filter buildings with missing percentage < threshold_pct
    valid_buildings = [bid for bid, pct in zip(df_readings.columns, missing_pcts) if pct < threshold_pct]
    
    # Sort the list of building IDs
    valid_buildings.sort()

    return {"answer": valid_buildings, "unit": None}
