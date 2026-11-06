"""EnergyX-Max program for template CS-001-T1.

What is the floor area of building {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get('building_id')
    
    if not building_id:
        raise ValueError("Missing required slot: building_id")
    
    try:
        # Retrieve metadata for the specific building
        meta_df = data.metadata()
        
        # The DataFrame is indexed by building_id. We look up the row for our target building.
        # If the building is not found in the index, it doesn't exist in the dataset.
        if building_id not in meta_df.index:
            raise ValueError(f"Building '{building_id}' not found in metadata.")
        
        # Extract the floor_area_m2 value from the row
        floor_area = meta_df.loc[building_id, 'floor_area_m2']
        
        # Return the result as a number (float) with unit square meters
        return {"answer": float(floor_area), "unit": "square meters"}
        
    except Exception as e:
        raise ValueError(f"Error retrieving floor area for {building_id}: {str(e)}")
