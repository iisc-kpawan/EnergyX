"""EnergyX-Max program for template CS-249-T2.

What is the total floor area of {building_type} buildings at the {site_id} site?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, **slots):
    building_type = slots.get("building_type")
    site_id = slots.get("site_id")
    if not building_type or not site_id:
        raise ValueError("Missing required slot values.")
    try:
        bldg_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError("No buildings found matching the criteria.")
    if not bldg_ids:
        raise ValueError("No buildings found matching the criteria.")
    meta = data.metadata()
    areas = meta.loc[bldg_ids, "floor_area_m2"]
    return {"answer": float(areas.sum()), "unit": "m2"}
