"""EnergyX-Max program for template EDA-Q046-T1.

What is the 5th percentile of hourly electricity consumption for all {building_type} buildings in {year}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import numpy as np
    
    building_type = slots['building_type']
    year = int(slots['year'])
    
    # Get list of building IDs for the specified building type
    try:
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found with type '{building_type}'")
        
    if not building_ids:
        raise ValueError(f"No buildings found with type '{building_type}' in year {year}")

    # Get start and end timestamps for the year
    try:
        start, end = data.year_range(year)
    except Exception:
        raise ValueError(f"Could not determine year range for {year}")

    # Read consumption data for these buildings
    try:
        readings_df = data.readings(building_ids, start=start, end=end)
    except Exception as e:
        raise ValueError(f"Error reading consumption data: {str(e)}")

    if readings_df.empty:
        raise ValueError("No consumption data found for the specified buildings and year.")

    # Filter out NaN values (invalid readings)
    valid_readings = readings_df.dropna()

    if valid_readings.empty:
        raise ValueError("No valid (non-NaN) hourly consumption data found for the specified buildings and year.")

    # Check qualifying criteria: at least 90% of hours valid per building
    total_hours = len(readings_df)
    min_valid_hours = int(total_hours * 0.9)
    
    # Count valid hours per building
    valid_counts = readings_df.count()
    qualifying_ids = [bid for bid, count in zip(readings_df.columns, valid_counts) if count >= min_valid_hours]

    if not qualifying_ids:
        raise ValueError(f"No buildings of type '{building_type}' met the 90% data completeness threshold for year {year}.")

    # Filter readings to only include qualifying buildings
    qualifying_readings = readings_df[qualifying_ids].dropna()

    if qualifying_readings.empty:
        raise ValueError("No valid consumption data remains after applying data completeness threshold.")

    # Flatten the DataFrame to a 1D array of all valid hourly values
    all_values = qualifying_readings.values.flatten()

    if len(all_values) == 0:
        raise ValueError("No valid hourly consumption data available for calculation.")

    # Calculate the 5th percentile
    try:
        result = np.percentile(all_values, 5)
    except Exception as e:
        raise ValueError(f"Error calculating percentile: {str(e)}")

    # Round to 6 decimals
    result = round(float(result), 6)

    return {"answer": result, "unit": "kWh"}
