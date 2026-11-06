"""EnergyX-Max program for template CS-248-T1.

How many {building_type} buildings are available at the {site_id} site in {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, **slots):
    btype = slots.get('building_type')
    site_id = slots.get('site_id')
    country = slots.get('country')
    
    if not btype or not site_id or not country:
        raise ValueError("Missing required slot values")
        
    try:
        bldgs = data.buildings(site_id=site_id, country=country)
    except Exception:
        raise ValueError(f"No buildings found for site {site_id} in {country}")
        
    if not bldgs:
        raise ValueError("No buildings found")
        
    df_meta = data.metadata()
    mask = (df_meta['building_type'] == btype) & (df_meta['site_id'] == site_id) & (df_meta['country'] == country)
    matching_ids = df_meta[mask].index.tolist()
    
    if not matching_ids:
        raise ValueError(f"No {btype} buildings found at site {site_id} in {country}")
        
    return {"answer": len(matching_ids), "unit": None}
