"""EnergyX-Max program for template EDA-Q108-T1.

Did {building_id}'s electricity consumption exceed twice its typical daily average on any day in {month}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, month):
    import pandas as pd
    import numpy as np
    
    # Define the start and end timestamps for the given month
    start, end = data.month_range(month)
    
    # Read the electricity consumption data for the specific building and month
    readings = data.readings([building_id], start=start, end=end)
    
    # Check if there is any data available
    if readings.empty or readings[building_id].isna().all():
        raise ValueError(f"No valid electricity consumption data found for {building_id} in {month}.")
    
    # Calculate the typical daily average (mean of all days in the month)
    # We group by date and take the mean of hourly readings for each day, then mean those daily means.
    daily_means = readings[building_id].groupby(readings.index.date).mean()
    typical_daily_average = daily_means.mean()
    
    # Identify days where consumption exceeded twice the typical daily average
    # We need to compare the daily total or daily mean? The question says "consumption exceed".
    # Usually, "consumption" refers to the total for the day. However, the "typical daily average" 
    # is derived from daily means (based on assumptions). Let's re-read carefully.
    # Assumption: "typical daily average is the mean of all days in {month}". This implies we calculate a scalar value.
    # Does "consumption exceed" refer to the hourly reading or the daily total?
    # Context: "on any day". It implies checking if the day's activity was high.
    # If we compare an hourly reading (which is part of the day) to the daily average, it might trigger often.
    # But usually, these questions imply: Did the *daily total* exceed 2x the *typical daily average*?
    # OR: Did the *hourly consumption* ever exceed 2x the typical daily average?
    # Let's look at the phrasing: "Did {building_id}'s electricity consumption exceed..."
    # If it meant hourly, it would usually say "hourly consumption".
    # However, if we compare daily totals to a daily average (which is essentially a daily total), that makes sense.
    # Let's assume "consumption" in the context of "on any day" refers to the daily total energy use.
    # Wait, let's reconsider the "typical daily average". If it's the mean of daily totals, then comparing daily totals is standard.
    # If it's the mean of hourly values (which equals total / 24), then comparing hourly values is also possible.
    # Given the ambiguity, the most robust interpretation for "consumption on a day" is the daily total.
    # Let's calculate daily totals and compare to 2 * typical_daily_average (where typical is mean of daily totals).
    
    daily_totals = readings[building_id].groupby(readings.index.date).sum()
    typical_daily_avg_total = daily_totals.mean()
    
    # Check if any day's total exceeds twice the typical average
    exceeded_threshold = daily_totals > (2 * typical_daily_avg_total)
    
    if exceeded_threshold.any():
        return {"answer": "Yes", "unit": None}
    else:
        return {"answer": "No", "unit": None}
