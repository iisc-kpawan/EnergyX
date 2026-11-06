"""EnergyX-Max program for template EDA-Q004-T1.

How many {building_type} buildings are available at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    site_id = slots.get("site_id")
    
    if not building_type or not site_id:
        raise ValueError("Missing required slot values: building_type and site_id.")

    try:
        # Get list of building IDs for the specified site
        building_ids = data.buildings(site_id=site_id)
        
        if not building_ids:
            raise ValueError(f"No buildings found at site '{site_id}'.")
            
        # Filter buildings by type
        # Note: The metadata DataFrame has 'building_type' column. 
        # We assume the slot value matches exactly what is in the dataframe.
        count = 0
        for bid in building_ids:
            meta = data.metadata().loc[bid]
            if meta['building_type'] == building_type:
                count += 1
                
        return {"answer": count, "unit": None}
        
    except Exception as e:
        raise ValueError(str(e))
