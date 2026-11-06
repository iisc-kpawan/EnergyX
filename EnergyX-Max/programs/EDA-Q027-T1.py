"""EnergyX-Max program for template EDA-Q027-T1.

Does electricity meter {building_id} contain any duplicated timestamps?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    building_id = slots.get("building_id")
    
    if not isinstance(building_id, str):
        raise ValueError("building_id must be a string.")
        
    try:
        df = data.readings([building_id])
    except Exception:
        raise ValueError(f"No data found for building {building_id}.")

    # Check if the index (timestamps) has any duplicates
    has_duplicates = df.index.has_duplicates
    
    if has_duplicates:
        return {"answer": "Yes", "unit": None}
    else:
        return {"answer": "No", "unit": None}
