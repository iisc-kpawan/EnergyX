"""EnergyX-Max program for template CS-080-T2.

Compare the average demand variability of {building_type} buildings versus {building_type_2} buildings across all sites.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    building_type_2 = slots.get("building_type_2")

    # Retrieve building IDs for each type
    bldg_ids_type1 = data.buildings(building_type=building_type)
    bldg_ids_type2 = data.buildings(building_type=building_type_2)

    if not bldg_ids_type1:
        raise ValueError(f"No buildings found with building_type '{building_type}'")
    if not bldg_ids_type2:
        raise ValueError(f"No buildings found with building_type '{building_type_2}'")

    # Determine the year range to cover all data for variability calculation
    # We assume we need the full available history. Let's get the min and max years from metadata if possible,
    # or just read a wide dataframe without date limits to infer the range.
    # However, the 'readings' function requires start/end or returns all if not specified? 
    # The docstring says: "start=None, end=None". Let's assume it returns all available data.
    
    try:
        # Get readings for both sets of buildings to determine the common time range
        df_all = data.readings(bldg_ids_type1 + bldg_ids_type2)
        
        if df_all.empty:
            raise ValueError("No reading data found for the specified building types.")

        start_time = df_all.index.min()
        end_time = df_all.index.max()
        # The 'end' parameter in readings is exclusive, but here we just pass None to get everything.
        # We need to ensure we are comparing over the same period. 
        # Since we want "across all sites", we use the full available history for both groups.
        
        # Filter data for each group within the common time range
        df_type1 = data.readings(bldg_ids_type1, start=start_time, end=end_time)
        df_type2 = data.readings(bldg_ids_type2, start=start_time, end=end_time)

    except Exception as e:
        raise ValueError(f"Error retrieving reading data: {str(e)}")

    # Calculate standard deviation for each building
    # Valid readings are non-NaN.
    
    try:
        std_dev_type1 = df_type1.std(axis=0).dropna()
        std_dev_type2 = df_type2.std(axis=0).dropna()
        
        # Ensure we have valid standard deviations (non-zero variance buildings)
        # If a building has constant consumption, std is 0. We include it as variability is 0.
        # However, if all readings are NaN for a building, dropna handles it.
        
        mean_std_type1 = std_dev_type1.mean()
        mean_std_type2 = std_dev_type2.mean()

        result = {
            "building_type": building_type,
            "avg_std_deviation_kwh": round(mean_std_type1, 6),
            "building_type_2": building_type_2,
            "avg_std_deviation_kwh": round(mean_std_type2, 6)
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating variability: {str(e)}")

    return {"answer": result, "unit": None}
