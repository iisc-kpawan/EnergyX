"""EnergyX-Max program for template EDA-Q065-T2.

For all {building_type} buildings in {city}, what is the average outdoor air temperature during hours when demand exceeds the 95th percentile of that year's peak?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    building_type = slots.get('building_type')
    city = slots.get('city')
    
    # Filter buildings by city and type
    try:
        building_ids = data.buildings(city=city, building_type=building_type)
    except Exception as e:
        raise ValueError(f"No buildings found for {building_type} in {city}.")

    if not building_ids:
        raise ValueError(f"No buildings found for {building_type} in {city}.")

    # Get readings for all selected buildings to determine the time range
    try:
        df_readings = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Error reading building data: {str(e)}")

    if df_readings.empty:
        raise ValueError("No readings available for the specified buildings.")

    # Determine the latest full year from the index to define the scope.
    # The problem implies "that year's peak", usually meaning the most recent full year available.
    last_date = df_readings.index.max()
    start_of_year = pd.Timestamp(last_date.year, 1, 1)
    end_of_year = pd.Timestamp(last_date.year + 1, 1, 1)
    
    # Filter data to the specific year
    mask = (df_readings.index >= start_of_year) & (df_readings.index < end_of_year)
    df_yearly = df_readings.loc[mask]

    if df_yearly.empty:
        raise ValueError("No full year of data available for the specified buildings.")

    # Calculate 95th percentile demand for each building individually
    percentiles = {}
    for bid in df_yearly.columns:
        valid_demand = df_yearly[bid].dropna()
        if len(valid_demand) == 0:
            continue
        p95 = valid_demand.quantile(0.95)
        percentiles[bid] = p95

    if not percentiles:
        raise ValueError("No valid demand data found to calculate percentiles.")

    # Get weather data for the same period. 
    # Since site_id is not provided in slots, we assume weather is available globally or per-site.
    # The API description says data.weather(site_id, ...). If site_id is None, it might fail if no global weather exists.
    # However, usually in these datasets, weather is tied to a site. 
    # Let's try to get site_ids from metadata for the buildings to fetch weather properly.
    
    try:
        # Get unique site_ids for the buildings
        site_ids = data.metadata().loc[building_ids, 'site_id'].unique()
        
        if len(site_ids) == 0:
            raise ValueError("No site_id found for the specified buildings.")
            
        # Fetch weather for each site and concatenate
        weather_dfs = []
        for sid in site_ids:
            try:
                df_w = data.weather(site_id=sid, start=start_of_year, end=end_of_year)
                if not df_w.empty:
                    weather_dfs.append(df_w)
            except Exception:
                continue
        
        if not weather_dfs:
             raise ValueError("No weather data available for the sites associated with the specified buildings.")
             
        df_weather = pd.concat(weather_dfs, axis=0).drop_duplicates(subset=['air_temperature'])
        
    except Exception as e:
        # Fallback: try fetching with None site_id if possible, or re-raise
        try:
            df_weather = data.weather(site_id=None, start=start_of_year, end=end_of_year)
            if df_weather.empty:
                raise ValueError("No weather data available for the specified period.")
        except Exception as e2:
             raise ValueError(f"No weather data available for the specified period. {str(e2)}")

    # Ensure alignment of timestamps. 
    # We need to merge readings and weather on index.
    
    # Create a boolean mask for each building indicating if demand > p95
    valid_hours_mask = pd.DataFrame(False, index=df_yearly.index, columns=percentiles.keys())
    
    for bid in percentiles:
        threshold = percentiles[bid]
        # Compare demand with threshold. Handle NaNs by treating them as not exceeding (False)
        exceeds = df_yearly[bid].notna() & (df_yearly[bid] > threshold)
        valid_hours_mask[bid] = exceeds

    # Identify unique timestamps where at least one building exceeds threshold
    valid_timestamps = df_yearly.index[valid_hours_mask.any(axis=1)]

    if len(valid_timestamps) == 0:
        raise ValueError("No hours found where demand exceeded the 95th percentile for any building.")

    # Get temperatures for these timestamps. 
    temps = []
    
    for ts in valid_timestamps:
        # Check if timestamp exists in weather index (handle timezone differences if any by using get_loc or direct check)
        if ts in df_weather.index:
            temp_val = df_weather.loc[ts, 'air_temperature']
            if pd.notna(temp_val):
                temps.append(temp_val)

    if not temps:
        raise ValueError("No valid temperature readings found during the specified hours.")

    avg_temp = np.mean(temps)

    return {"answer": round(avg_temp, 6), "unit": "degC"}
