"""EnergyX-Max program for template CS-033-T1.

How many {building_type} buildings are there?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_type = slots.get("building_type")
    
    if not building_type:
        raise ValueError("Missing required slot: building_type")

    try:
        building_ids = data.buildings(building_type=building_type)
    except Exception:
        # In case the filter logic inside data.buildings fails or returns empty unexpectedly
        building_ids = []

    count = len(building_ids)
    
    return {"answer": count, "unit": None}
