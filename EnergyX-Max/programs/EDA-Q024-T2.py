"""EnergyX-Max program for template EDA-Q024-T2.

What is the total hourly electricity consumption for {building_id} in {year} after imputing missing data?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, building_id, year):
    import numpy as np
    
    # Retrieve the specific building ID from the list of all buildings to ensure it exists and is valid
    try:
        building_ids = data.buildings()
        if building_id not in building_ids:
            raise ValueError(f"Building ID '{building_id}' not found in the dataset.")
        
        # Get the start and end timestamps for the specified year
        start, end = data.year_range(year)
        
        # Read the hourly electricity consumption data for the specific building
        readings_df = data.readings(building_ids=[building_id], start=start, end=end)
        
        if readings_df.empty:
            raise ValueError(f"No data found for building '{building_id}' in year {year}.")
        
        # Extract the single column for the building (it should be named by the building_id or similar)
        # The read function returns a wide DataFrame where columns are building IDs.
        series = readings_df[building_id]
        
        if series.empty:
            raise ValueError(f"No data found for building '{building_id}' in year {year}.")
            
        # Impute missing data using forward fill (ffill) followed by backward fill (bfill).
        # This is a standard method to handle gaps at the beginning or end of the series or small gaps.
        # If there are large gaps, linear interpolation might be better, but ffill/bfill is robust for 
        # typical utility data where missingness is often due to meter downtime at start/end or short periods.
        # Given "standard method", ffill+bfill is very common. Alternatively, linear interpolation (interpolate())
        # is also standard. Let's use linear interpolation as it preserves trends better than step functions.
        imputed_series = series.interpolate(method='linear')
        
        # Sum the total consumption for the year
        total_consumption = imputed_series.sum()
        
        return {"answer": round(float(total_consumption), 6), "unit": "kWh"}
        
    except Exception as e:
        raise ValueError(str(e))
