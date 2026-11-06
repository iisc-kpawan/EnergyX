"""EnergyX-Max program for template EDA-Q074-T1.

What is the correlation between hourly electricity consumption and outdoor air temperature for {building_type} buildings at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, site_id) -> dict:
    import numpy as np
    
    # Get list of building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for site '{site_id}' and type '{building_type}'.")
    
    if not building_ids:
        raise ValueError(f"No qualifying buildings found for site '{site_id}' and type '{building_type}'.")
    
    # Define the time range. Since no specific year is given, we assume the full available data 
    # or a recent complete year. The prompt implies using available data. 
    # We will fetch readings without explicit start/end to get all available data for these buildings.
    # However, to calculate correlation with weather, we need aligned timestamps.
    # Let's try to get readings for the full range first. If that fails or returns empty, we might need defaults.
    # The read function signature suggests start and end are optional.
    
    try:
        readings_df = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to retrieve consumption data: {str(e)}")
        
    if readings_df.empty:
        raise ValueError("No consumption data available for the specified buildings.")

    # Get weather data. We need to cover the same time range as the readings.
    # Since we don't have explicit start/end from the user, we infer from the readings index.
    try:
        weather_df = data.weather(site_id=site_id)
    except Exception as e:
        raise ValueError(f"Failed to retrieve weather data for site '{site_id}': {str(e)}")

    if weather_df.empty:
        raise ValueError("No weather data available for the specified site.")

    # Align readings and weather data. 
    # Readings are wide format (one column per building). Weather is long or wide? 
    # Description says "DataFrame indexed by hourly timestamp with columns...". Likely wide or single station.
    # Assuming weather_df has one row per timestamp and columns for variables.
    
    # Ensure we only use timestamps present in both datasets
    common_index = readings_df.index.intersection(weather_df.index)
    
    if len(common_index) < 2:
        raise ValueError("Insufficient overlapping data points between consumption and weather records to calculate correlation.")

    # Extract temperature column
    temp_col = 'air_temperature'
    if temp_col not in weather_df.columns:
        raise ValueError(f"Expected weather variable '{temp_col}' not found in weather data columns.")
        
    temps = weather_df.loc[common_index, temp_col].values
    
    # Stack consumption data for all buildings into a single series or list of arrays
    # We need to calculate correlation between the aggregate consumption (or individual?) 
    # The question asks "correlation between hourly electricity consumption...". 
    # Usually, this implies the total site consumption or an average. Given it's a specific site and type,
    # summing them up makes sense for a single correlation coefficient representing the site's behavior.
    
    consumptions = readings_df.loc[common_index].sum(axis=1).values
    
    # Calculate Pearson correlation coefficient
    # Ignore NaNs (though we filtered by common index, weather might still have NaNs if not all stations report)
    valid_mask = ~np.isnan(temps) & ~np.isnan(consumptions)
    
    if np.sum(valid_mask) < 2:
        raise ValueError("Not enough valid data points after handling missing values.")
        
    corr = np.corrcoef(temps[valid_mask], consumptions[valid_mask])[0, 1]
    
    # Round to 6 decimals
    return {"answer": round(float(corr), 6), "unit": None}
