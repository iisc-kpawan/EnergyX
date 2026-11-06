"""EnergyX-Max program for template CS-249-T1.

Which sites contain {building_type} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd

def answer(data, **slots):
    building_type = slots.get("building_type")
    if not building_type:
        raise ValueError("Missing building_type slot")
    try:
        df = data.metadata()
    except Exception:
        raise ValueError("Cannot access metadata")
    mask = df["building_type"].str.lower().str.strip() == building_type.lower().strip()
    sites = set(df.loc[mask, "site_id"])
    if not sites:
        return {"answer": [], "unit": None}
    return {"answer": sorted(sites), "unit": None}
