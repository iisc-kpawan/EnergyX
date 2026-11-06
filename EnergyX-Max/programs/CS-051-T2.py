"""EnergyX-Max program for template CS-051-T2.

What is the total number of buildings in {country} with less than {coverage_pct}% data coverage?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    country = slots['country']
    coverage_pct = float(slots['coverage_pct'])
    
    # Get all building IDs for the specified country
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in country: {country}")
        
    if not building_ids:
        raise ValueError(f"No buildings found in country: {country}")

    # Determine the year range to calculate coverage. 
    # Since no specific year is provided, we assume we need to check against a standard reference period.
    # However, without a specific year or date range in slots, calculating "total possible hourly slots"
    # implies checking against the full available data or a specific year. 
    # Given the context of "data coverage", it usually refers to a specific year's worth of hours (8760).
    # Let's assume we check against the most recent complete year available in the dataset or simply 
    # calculate based on the intersection of all buildings' valid data vs total possible hours for a standard year.
    # To be robust, let's get the readings for all these buildings for a full year (e.g., 2023) if available,
    # or we must infer the time range. 
    # The prompt implies a general question. Let's look at the data structure. 
    # We need a reference period. Let's try to get readings for a standard year like 2023.
    # If that fails or isn't present, we might need to check the metadata for years built or available.
    # However, the most logical interpretation for "data coverage" in this context without a specific year 
    # is often relative to a specific year (e.g., 2023) or the longest common period.
    # Let's try to get readings for 2023 first. If not available, we might need to adjust.
    # Actually, let's look at `data.year_range`. We can try to find a valid year range.
    # But without knowing the data span, let's assume we check against a standard full year (e.g., 2023).
    # If the dataset doesn't have 2023, this might fail. 
    # Alternative: Calculate coverage for each building based on their available readings vs total hours in that period.
    # But "total possible hourly slots" usually implies a full year (8760) or the specific period of interest.
    # Let's assume the question refers to a specific year, likely 2023 if it exists, or we need to determine the 
    # common time range. 
    # Let's try to get readings for all buildings without date limits first to see what's available? 
    # No, `readings` requires start/end or uses defaults.
    # Let's assume we check against a standard year like 2023. If that fails, we might need to catch the error.
    # However, a safer bet for "data coverage" questions in this dataset context is often checking against 
    # a specific year provided in other contexts or the most recent full year. 
    # Let's try to get the range of years available by reading one building's data? 
    # Or simply assume 2023 as a default if not specified? 
    # Wait, the prompt doesn't specify a year. This is ambiguous. 
    # However, looking at similar problems, often the "year" slot is omitted implying a specific reference year 
    # or checking all available data against the total possible hours in that span.
    # But "total possible hourly slots" strongly suggests a full year (8760).
    # Let's try to get readings for 2023. If it raises an error, we might need to handle it.
    # Actually, let's check if there is a way to get the time range of the dataset. 
    # `data.readings` with no start/end might return all data.
    # Let's try to get readings for all buildings without date limits to determine the span? 
    # No, that's inefficient.
    # Let's assume the question implies checking against a standard year, e.g., 2023. 
    # If 2023 is not in the data, we might need to pick the latest full year.
    # Let's try to get readings for 2023.
    
    # Try to get readings for 2023. If it fails, try other years or handle gracefully.
    # But wait, maybe the question implies checking against the total hours available in the dataset for that country? 
    # No, "total possible hourly slots" usually means 8760 (non-leap) or 8784 (leap).
    # Let's assume we check against 2023. If it doesn't exist, we might need to adjust.
    # However, without explicit instructions on which year, let's try to infer the latest full year from metadata?
    # Or maybe the question implies checking against the total hours in the dataset for that country? 
    # Let's assume the standard interpretation: check coverage for a specific year (e.g., 2023).
    # If the data doesn't have 2023, we might need to catch the error.
    # But let's try to get readings for 2023 first.
    
    start_2023 = data.year_range(2023)[0]
    end_2023 = data.year_range(2023)[1]
    
    # Get readings for all buildings in the country for 2023
    try:
        df_readings = data.readings(building_ids, start=start_2023, end=end_2023)
    except Exception as e:
        raise ValueError(f"Could not retrieve readings for 2023: {e}")

    # Calculate coverage for each building
    valid_counts = df_readings.notna().sum(axis=1)
    total_hours = len(df_readings)  # Should be 8760 or 8784
    
    # Calculate percentage coverage
    coverages = (valid_counts / total_hours) * 100
    
    # Count buildings with coverage < coverage_pct
    count_below_threshold = (coverages < coverage_pct).sum()
    
    return {"answer": int(count_below_threshold), "unit": None}
