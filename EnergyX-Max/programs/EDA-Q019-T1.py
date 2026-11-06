"""EnergyX-Max program for template EDA-Q019-T1.

How many duplicate timestamps exist in the weather dataset for the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    site_id = slots.get("site_id")
    
    # Retrieve weather data for the specified site
    # The function returns a DataFrame indexed by hourly timestamp.
    try:
        weather_df = data.weather(site_id=site_id)
    except Exception:
        raise ValueError(f"No weather data found for site '{site_id}'.")

    if weather_df.empty:
        raise ValueError(f"No weather data found for site '{site_id}'.")

    # Get the index (timestamps) of the weather dataset
    timestamps = weather_df.index
    
    # Count duplicate timestamps. 
    # value_counts() counts occurrences of each timestamp.
    # summing the values where count > 1 gives the number of entries that are duplicates.
    # Alternatively, len(timestamps) - len(unique_timestamps) gives the count of extra rows.
    unique_count = timestamps.nunique()
    
    # Total number of rows
    total_count = len(timestamps)
    
    # Number of duplicate entries (rows that share a timestamp with at least one other row)
    # If a timestamp appears 3 times, it contributes 2 to the count of "duplicate" rows.
    duplicate_timestamps_count = total_count - unique_count

    return {"answer": duplicate_timestamps_count, "unit": None}
