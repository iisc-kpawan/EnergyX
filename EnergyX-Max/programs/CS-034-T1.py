"""EnergyX-Max program for template CS-034-T1.

How many {building_type} buildings are included in the dataset?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    
    if not building_type:
        raise ValueError("Missing required slot: building_type")
        
    try:
        # Get list of building IDs matching the specified building type
        building_ids = data.buildings(building_type=building_type)
        
        # Check if any buildings were found
        if not building_ids:
            raise ValueError(f"No buildings found with type '{building_type}'")
            
        return {"answer": len(building_ids), "unit": None}
        
    except Exception as e:
        raise ValueError(str(e))
