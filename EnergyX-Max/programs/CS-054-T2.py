"""EnergyX-Max program for template CS-054-T2.

For buildings of type {building_type} in {city}, what is the average percentage of hours where consumption exceeds the 95th percentile of all readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd

def answer(data, building_type, city):
    # Filter buildings by city and type
    try:
        building_ids = data.buildings(city=city, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for type '{building_type}' in city '{city}'.")

    if not building_ids:
        raise ValueError(f"No buildings found for type '{building_type}' in city '{city}'.")

    # Get readings for all selected buildings
    try:
        df = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to retrieve readings: {str(e)}")

    if df.empty:
        raise ValueError("No valid readings found for the specified buildings.")

    # Flatten the DataFrame to a Series (building_id, timestamp -> consumption)
    series = df.stack()
    
    # Filter out NaN values immediately as they are invalid
    series_valid = series.dropna()

    if series_valid.empty:
        raise ValueError("No valid readings found after filtering NaNs.")

    # Calculate the 95th percentile across ALL valid readings
    threshold_95 = np.percentile(series_valid.values, 95)

    # Count total valid hours and hours exceeding the threshold
    total_hours = len(series_valid)
    if total_hours == 0:
        raise ValueError("No valid hours found to calculate statistics.")

    hours_exceeding = (series_valid > threshold_95).sum()

    # Calculate percentage for this specific set of data points
    # The question asks for the average percentage. Since we are aggregating across all buildings 
    # and all their hours, calculating the percentage over the entire valid dataset is the standard interpretation 
    # for "average percentage of hours" in this context (global proportion).
    # If the intention was to average the percentages per building first, then average those, 
    # the result would be slightly different if buildings have different numbers of valid hours.
    # However, usually "average percentage of hours where..." implies the global ratio unless specified "average of the percentages".
    # Let's re-read carefully: "average percentage of hours". 
    # Interpretation A: Sum(exceeding) / Sum(total) -> Global %.
    # Interpretation B: Mean( (exceeding_i / total_i) for each building i ).
    # Given the phrasing "average percentage of hours", Interpretation A is the most robust statistical measure 
    # of the frequency of high consumption events across the population. 
    # However, if we look at similar energy questions, often they want the mean of the per-building percentages.
    # Let's stick to the global calculation (Interpretation A) as it represents the actual percentage of hours in the dataset.
    # Wait, let's reconsider. If I have Building A with 100 hours (5 exceed) and Building B with 10 hours (9 exceed).
    # Global: 14/110 = 12.7%.
    # Mean of per-building: (5% + 90%) / 2 = 47.5%.
    # The prompt says "average percentage of hours". This usually implies the mean of the percentages if the population is buildings.
    # But strictly speaking, "percentage of hours" is a property of time. 
    # Let's assume the question asks for the mean of the per-building percentages to account for building-level performance.
    
    # Recalculating based on per-building average:
    percentages = []
    for bid in building_ids:
        col_data = df[bid].dropna()
        if len(col_data) == 0:
            continue
        pct = (col_data > threshold_95).sum() / len(col_data) * 100
        percentages.append(pct)

    if not percentages:
        raise ValueError("No valid hours found for any building.")

    avg_percentage = np.mean(percentages)

    return {"answer": round(avg_percentage, 6), "unit": "%"}
