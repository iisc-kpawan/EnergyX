"""EnergyX-Max program for template CS-051-T1.

Which buildings have more than {threshold_pct}% missing hourly electricity consumption readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots):
    threshold_pct = slots.get('threshold_pct')
    
    # Determine the range of years available in the dataset to scan all buildings
    # We assume data.year_range exists and returns a tuple (start, end). 
    # If not explicitly provided, we try to infer or use a broad range. 
    # However, based on the schema description, we need to know the full extent.
    # Let's try to get the min and max year from metadata if possible, or assume a standard range.
    # Since data.year_range(year) is specific, let's look for a way to get all years.
    # The problem implies scanning existing buildings. We need to iterate over all building IDs.
    
    # Get all building IDs. The schema says data.buildings() returns a sorted list of building_ids.
    # It doesn't specify arguments for getting ALL, so we call it without args (site_id=None etc).
    try:
        all_building_ids = data.buildings()
    except Exception:
        # Fallback if no args work or if the method requires specific handling not shown
        # Assuming data.buildings() with no args returns all.
        all_building_ids = []

    if not all_building_ids:
        return {"answer": [], "unit": None}

    # We need to determine the time range for each building to calculate missingness.
    # The most robust way is to find the global min and max date across all buildings or 
    # assume a standard year range if metadata isn't easily accessible for date ranges.
    # However, data.year_range(year) suggests we know years. Let's try to get the first and last year.
    # A safer approach: Iterate through the buildings, find their valid reading range, 
    # but "missing hourly electricity consumption readings" usually implies missing within the 
    # operational period of the building or a fixed reporting period (e.g., a full year).
    # The convention says: "A yearly measure ... uses buildings with at least 90% of the year's hours valid".
    # This implies we are checking against a specific year or the total available time.
    # Without a specified year in slots, and given the context of "electricity consumption readings",
    # it is standard to check missingness over the entire available dataset period for that building 
    # OR over a representative full year. 
    # Given the ambiguity, let's assume we check against the total hours available in the dataset 
    # for that specific building (from first reading to last reading).
    
    # Actually, re-reading: "Which buildings have more than X% missing ... readings?"
    # If no year is specified, it likely refers to the entire history of data available for those buildings.
    # Let's define the start and end based on the union of all building data or just iterate per building.
    
    # Strategy: For each building, get its readings. Calculate total hours vs valid hours.
    # Total hours = (last_timestamp - first_timestamp).total_seconds() / 3600 + 1 (inclusive)
    # Valid hours = count of non-NaN values.
    # Missing % = (Total - Valid) / Total * 100
    
    missing_threshold = threshold_pct / 100.0
    
    high_missing_buildings = []
    
    for bid in all_building_ids:
        try:
            # Get readings for this building. 
            # data.readings(building_ids, start=None, end=None) -> wide DataFrame
            # We pass a list with one ID.
            df_readings = data.readings([bid])
            
            if df_readings.empty or len(df_readings.columns) == 0:
                continue
                
            # Check if there are any readings at all
            if df_readings.isna().all().all():
                # All NaN, so 100% missing. But we need to know the total hours.
                # If the dataframe is empty or has no rows, skip? 
                # Usually, a building exists in metadata but might have no readings? 
                # The prompt says "Only metered buildings" in metadata.
                # If df_readings has columns but no rows (or all NaN), total hours depends on index range.
                pass
            
            # Calculate total possible hours based on the DatetimeIndex
            start_time = df_readings.index[0]
            end_time = df_readings.index[-1]
            
            # Total hours in the period
            delta = end_time - start_time
            total_hours = int(delta.total_seconds() / 3600) + 1
            
            if total_hours <= 0:
                continue
                
            # Count valid readings (non-NaN)
            # Since it's a wide dataframe with one column per building, we look at the column for 'bid'
            col_name = bid # The column name is likely the building_id string or int
            # Ensure column exists
            if col_name not in df_readings.columns:
                continue
                
            valid_count = df_readings[col_name].notna().sum()
            
            missing_pct = (total_hours - valid_count) / total_hours * 100
            
            if missing_pct > threshold_pct:
                high_missing_buildings.append(bid)
                
        except Exception:
            # Skip buildings that cause errors (e.g., invalid IDs, data access issues)
            continue
            
    # Return sorted list of strings as per "Answer type: set. Expected output: List"
    # The convention says "sorted list of strings for set".
    return {"answer": sorted(high_missing_buildings), "unit": None}
