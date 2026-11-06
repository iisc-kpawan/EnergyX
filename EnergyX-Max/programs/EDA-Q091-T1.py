"""EnergyX-Max program for template EDA-Q091-T1.

How many buildings are located at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    site_id = slots.get("site_id")
    
    if not site_id:
        raise ValueError("Missing site_id slot.")
        
    try:
        building_ids = data.buildings(site_id=site_id)
    except Exception:
        raise ValueError(f"No buildings found at site '{site_id}'.")
        
    count = len(building_ids)
    
    return {"answer": count, "unit": None}
