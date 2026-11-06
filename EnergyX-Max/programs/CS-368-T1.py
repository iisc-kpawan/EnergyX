"""EnergyX-Max program for template CS-368-T1.

Which {top_n} buildings have load profiles most similar to {building_id}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np

def answer(data, **slots):
    building_id = slots['building_id']
    top_n = slots['top_n']
    try:
        readings_all = data.readings([building_id])
        if building_id not in readings_all.columns or readings_all[building_id].isna().all():
            raise ValueError("Reference building has no valid load data.")
        ref_load = readings_all[building_id].dropna()
        median_ref = ref_load.median()
        ref_norm = (ref_load - median_ref) / median_ref
        
        candidates = data.buildings()
        candidate_ids = [bid for bid in candidates if bid != building_id]
        
        if not candidate_ids:
            return {"answer": [], "unit": None}
            
        readings_cands = data.readings(candidate_ids)
        valid_cands = []
        for bid, col in readings_cands.items():
            if col.isna().all(): continue
            c_load = col.dropna()
            if len(c_load) < len(ref_load): continue
            median_c = c_load.median()
            c_norm = (c_load - median_c) / median_c
            min_len = min(len(ref_norm), len(c_norm))
            dist = np.sqrt(np.mean((ref_norm[:min_len] - c_norm[:min_len])**2))
            valid_cands.append((bid, dist))
        
        if not valid_cands:
            raise ValueError("No comparable buildings found.")
            
        valid_cands.sort(key=lambda x: x[1])
        return {"answer": [x[0] for x in valid_cands[:top_n]], "unit": None}
    except Exception as e:
        raise ValueError(str(e))
