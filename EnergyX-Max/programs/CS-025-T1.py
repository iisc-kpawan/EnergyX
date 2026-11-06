"""EnergyX-Max program for template CS-025-T1.

How many buildings are located in {city}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    city = slots.get("city")
    
    if not isinstance(city, str):
        raise ValueError("City must be a string.")
        
    try:
        building_ids = data.buildings(city=city)
    except Exception:
        raise ValueError(f"No buildings found in city '{city}'.")
        
    if not building_ids:
        raise ValueError(f"No buildings found in city '{city}'.")
        
    return {"answer": len(building_ids), "unit": None}
