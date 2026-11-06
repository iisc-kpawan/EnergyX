"""EnergyX-Max program for template EDA-Q084-T2.

What is the total hourly electricity consumption for {building_id} on {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import pandas as pd
    
    building_id = slots['building_id']
    date_str = slots['date']
    
    # Parse the date string to a datetime object
    try:
        target_date = pd.to_datetime(date_str)
    except Exception:
        raise ValueError(f"Invalid date format: {date_str}")

    # Define the start and end timestamps for the specific day
    # The question asks for consumption "on" that date, so we look at the 24 hours starting from midnight.
    start_ts = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = start_ts + pd.Timedelta(days=1) # Exclusive end time for the day

    try:
        # Retrieve readings for the specific building and date range
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
        
        if readings_df.empty or building_id not in readings_df.columns:
            raise ValueError(f"No data found for building '{building_id}' on date '{date_str}'.")

        # Get the column for the specific building
        col_name = building_id
        
        # Sum valid (non-NaN) readings for the day
        total_consumption = readings_df[col_name].sum()

        if pd.isna(total_consumption):
            raise ValueError(f"No valid consumption data found for building '{building_id}' on date '{date_str}'.")

        return {"answer": round(float(total_consumption), 6), "unit": "kWh"}
        
    except Exception as e:
        # Re-raise or wrap if necessary, but the prompt says raise ValueError with short reason
        raise ValueError(f"Error retrieving data for building '{building_id}' on date '{date_str}': {str(e)}")
