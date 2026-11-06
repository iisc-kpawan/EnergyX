"""EnergyX-Max program for template CS-032-T1.

How many {building_type} buildings are included in the entire dataset?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    
    if not building_type:
        raise ValueError("Missing required slot: building_type")

    try:
        # Get list of building_ids for the specified building type
        # The metadata function returns a DataFrame indexed by building_id.
        # We filter this dataframe by building_type and get the index (building_ids).
        metadata_df = data.metadata()
        
        # Filter for the specific building type
        filtered_metadata = metadata_df[metadata_df["building_type"] == building_type]
        
        if filtered_metadata.empty:
            raise ValueError(f"No buildings found with type '{building_type}' in the dataset.")
            
        # Get the list of building IDs
        building_ids = filtered_metadata.index.tolist()
        
        # The question asks for the count of such buildings included in the entire dataset.
        # Since we are filtering the metadata which represents the "entire dataset" (metered buildings),
        # the length of this list is the answer.
        count = len(building_ids)
        
        return {"answer": count, "unit": None}
        
    except Exception as e:
        raise ValueError(str(e))
