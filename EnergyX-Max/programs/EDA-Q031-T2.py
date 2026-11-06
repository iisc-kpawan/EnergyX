"""EnergyX-Max program for template EDA-Q031-T2.

Compare the mean hourly demand of {building_type} buildings in {city} versus {country} using data from {year}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_type, city, country, year):
    import numpy as np
    import pandas as pd
    
    # Filter buildings by type and location criteria
    try:
        # Get buildings matching the type in the specified city
        b_ids_city = data.buildings(city=city, building_type=building_type)
        
        # Get buildings matching the type in the specified country
        b_ids_country = data.buildings(country=country, building_type=building_type)
        
        if not b_ids_city and not b_ids_country:
            raise ValueError("No buildings found matching the criteria.")
            
    except Exception as e:
        # Fallback logic if direct filtering fails or behaves unexpectedly
        b_ids_all_type = data.buildings(building_type=building_type)
        
        # Filter by city manually
        b_ids_city = [bid for bid in b_ids_all_type if data.metadata()[bid].get('city') == city]
        # Filter by country manually
        b_ids_country = [bid for bid in b_ids_all_type if data.metadata()[bid].get('country') == country]
        
        if not b_ids_city and not b_ids_country:
            raise ValueError("No buildings found matching the criteria.")

    # Determine the time range for the year
    start_ts, end_ts = data.year_range(year)
    
    # Read energy data for both sets of buildings
    try:
        df_city = data.readings(b_ids_city, start=start_ts, end=end_ts)
        df_country = data.readings(b_ids_country, start=start_ts, end=end_ts)
    except Exception as e:
        raise ValueError(f"Error reading energy data: {str(e)}")
    
    # Check for empty datasets after filtering (e.g., if no buildings qualified or no data)
    if df_city.empty and df_country.empty:
        raise ValueError("No valid hourly demand data found for the specified buildings and year.")
        
    # Calculate mean hourly demand for each group
    # The read function returns a wide DataFrame with building_ids as columns.
    # We need to handle NaNs (missing data) by ignoring them in the mean calculation.
    
    if not df_city.empty:
        total_kwh_city = df_city.sum().sum()
        total_hours_city = df_city.count().sum() # Sum of counts (ignoring NaNs)
        if total_hours_city > 0:
            mean_hourly_city = total_kwh_city / total_hours_city
        else:
            mean_hourly_city = np.nan
            
    else:
        mean_hourly_city = np.nan

    if not df_country.empty:
        total_kwh_country = df_country.sum().sum()
        total_hours_country = df_country.count().sum()
        if total_hours_country > 0:
            mean_hourly_country = total_kwh_country / total_hours_country
        else:
            mean_hourly_country = np.nan
    else:
        mean_hourly_country = np.nan

    # Prepare the record
    result = {}
    
    if not pd.isna(mean_hourly_city):
        result['city_mean'] = round(float(mean_hourly_city), 6)
    else:
        result['city_mean'] = None
        
    if not pd.isna(mean_hourly_country):
        result['country_mean'] = round(float(mean_hourly_country), 6)
    else:
        result['country_mean'] = None

    # Add unit
    return {"answer": result, "unit": "kWh"}
