"""EnergyX-Max program for template CS-045-T1.

For building {building_id}, how many hourly timestamps have missing electricity consumption values or duplicate readings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    # Retrieve readings for the specific building
    # The read_data function returns a wide DataFrame where columns are building IDs
    df = data.readings([building_id])
    
    if df.empty:
        raise ValueError(f"No data found for building {building_id}.")

    # Get the column name for this building (it should be exactly the building_id)
    col_name = building_id
    
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in readings.")

    series = df[col_name]
    
    # Count missing values (NaN)
    missing_count = series.isna().sum()
    
    # Count duplicate consecutive rows. 
    # A row is a duplicate if it is identical to the previous one.
    # We check if current value equals previous value.
    # The first row cannot be a duplicate of a previous row.
    duplicates_count = 0
    for i in range(1, len(series)):
        if series.iloc[i] == series.iloc[i-1]:
            duplicates_count += 1
            
    total_problematic = missing_count + duplicates_count
    
    return {"answer": total_problematic, "unit": None}
