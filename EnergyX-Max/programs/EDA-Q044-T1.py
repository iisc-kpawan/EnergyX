"""EnergyX-Max program for template EDA-Q044-T1.

What is the average hourly electricity consumption for {building_type} buildings on weekdays versus weekends at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    site_id = slots['site_id']
    
    # Get list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found at site '{site_id}' with type '{building_type}'.")

    if not building_ids:
        raise ValueError(f"No buildings found at site '{site_id}' with type '{building_type}'.")

    # Define the date range for 2016-2017
    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2018-01-01') # Exclusive end
    
    try:
        readings = data.readings(building_ids, start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Error reading data for buildings {building_ids}: {str(e)}")

    if readings.empty:
        raise ValueError("No valid hourly readings found for the specified period.")

    # Filter for weekdays and weekends
    # pd.Timestamp.weekday returns 0-4 for Mon-Fri, 5-6 for Sat-Sun
    is_weekday = readings.index.weekday < 5
    is_weekend = ~is_weekday

    # Calculate total consumption per day for each building to aggregate properly
    # We need daily totals first to handle the "average hourly" correctly across days.
    # However, the question asks for average hourly consumption. 
    # Usually, this implies (Sum of kWh in period) / (Total hours in period).
    # But distinguishing weekdays vs weekends implies we calculate:
    # Avg_Hourly_Weekday = Sum(Consumption on Weekdays) / Count(Hours on Weekdays)
    # Avg_Hourly_Weekend = Sum(Consumption on Weekends) / Count(Hours on Weekends)
    
    # Let's sum consumption for weekdays and weekends separately across all buildings
    weekday_consumption = readings.loc[is_weekday].sum(axis=1)
    weekend_consumption = readings.loc[is_weekend].sum(axis=1)
    
    # Count hours (valid readings are used, but usually we count the time slots available)
    # The convention says: "A reading is valid when it is not NaN. Sums and means use valid readings only."
    # So we should sum valid readings and divide by count of valid readings.
    
    weekday_valid_count = readings.loc[is_weekday].notna().sum(axis=1)
    weekend_valid_count = readings.loc[is_weekend].notna().sum(axis=1)

    # Calculate averages
    avg_weekday = (weekday_consumption / weekday_valid_count).mean()
    avg_weekend = (weekend_consumption / weekend_valid_count).mean()

    # Prepare the record
    result = {
        "average_hourly_weekday": round(float(avg_weekday), 6),
        "average_hourly_weekend": round(float(avg_weekend), 6)
    }

    return {"answer": result, "unit": "kWh"}
