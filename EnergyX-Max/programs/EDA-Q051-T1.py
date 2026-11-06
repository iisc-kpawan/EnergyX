"""EnergyX-Max program for template EDA-Q051-T1.

What percentage of {building_id}'s weekly electricity consumption occurs outside typical business hours (7am-7pm, weekdays)?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    building_id = slots['building_id']
    
    # Retrieve readings for the specific building
    # We need a wide dataframe or just the single column. 
    # data.readings returns a wide DF with one column per building.
    try:
        df_readings = data.readings([building_id])
    except Exception:
        raise ValueError(f"Building {building_id} not found in metadata.")

    if building_id not in df_readings.columns:
        raise ValueError(f"Building {building_id} not found in readings.")

    # Get the specific column for this building
    col_name = building_id
    series = df_readings[col_name]

    # Determine the timezone of the building to handle local time correctly
    try:
        meta_df = data.metadata()
        site_id = meta_df.loc[building_id, 'site_id']
        tz_str = meta_df.loc[building_id, 'timezone']
        
        # If timezone is None or empty string, assume UTC and convert if necessary, 
        # but usually BDG2 has specific timezones. Let's try to localize.
        import pandas as pd
        
        # Create a datetime index from the series
        dti = series.index
        
        # Try to localize based on the site's timezone
        if pd.notna(tz_str) and tz_str:
            try:
                # Handle common timezone string formats (e.g., 'America/New_York')
                # If it's an offset like '+01:00', pandas might need pytz or zoneinfo.
                # Assuming standard IANA names or offsets supported by pd.date_range logic if we were creating one,
                # but here we have existing timestamps. 
                # Let's assume the index is UTC and convert to local time for filtering.
                from dateutil import tz
                local_tz = tz.gettz(tz_str)
                dti_local = dti.tz_localize('UTC').tz_convert(local_tz)
            except Exception:
                # Fallback if timezone string is weird or conversion fails, assume UTC for calculation 
                # or just use the index as is (assuming it's already local or UTC consistent).
                # Given the context of "local time", we must convert.
                # If tz_str is 'UTC', no conversion needed.
                if str(tz_str) == 'UTC':
                    dti_local = dti
                else:
                    raise ValueError(f"Could not localize timezone for {building_id}: {tz_str}")
        else:
            # If no timezone info, assume the timestamps are already in local time or UTC.
            # Without specific TZ, we cannot strictly define "local" 7am-7pm relative to a location 
            # unless we assume UTC and the user implies UTC business hours (unlikely) or the data is local.
            # However, standard practice for these datasets usually includes timezone.
            # If missing, we might have to assume UTC or raise an error. 
            # Let's assume UTC if no TZ provided to avoid crashing, but note this limitation.
            dti_local = dti.tz_localize('UTC') 

    except Exception:
        raise ValueError("Could not determine timezone for building.")

    # Define business hours mask: Weekdays (Mon-Fri) between 07:00 and 19:00
    # Note: 19:00 is exclusive in the prompt "7am-7pm", usually meaning up to but not including 19:00.
    # Let's interpret "outside typical business hours" as NOT (Weekday AND 07:00 <= hour < 19:00).
    
    # Create a boolean mask for business hours
    is_weekday = dti_local.dayofweek.isin([0, 1, 2, 3, 4]) # Monday=0 to Friday=4
    is_business_hour = (dti_local.hour >= 7) & (dti_local.hour < 19)
    
    business_hours_mask = is_weekday & is_business_hour
    
    # Calculate total consumption (sum of valid readings)
    total_consumption = series.sum()
    
    if pd.isna(total_consumption):
        raise ValueError("No electricity consumption data found for the specified building.")

    # Calculate consumption outside business hours
    # Sum where NOT business_hours_mask
    outside_business_hours_mask = ~business_hours_mask
    outside_consumption = series[outside_business_hours_mask].sum()
    
    if pd.isna(outside_consumption):
        outside_consumption = 0.0

    # Calculate percentage
    percentage = (outside_consumption / total_consumption) * 100
    
    # Round to 6 decimals
    return {"answer": round(percentage, 6), "unit": "%"}
