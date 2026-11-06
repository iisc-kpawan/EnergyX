"""EnergyX-Max program for template CS-251-T2.

Using K-Means with {k} clusters, identify the top {top_n} building-use types by total consumption across all buildings in {country}.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans

def answer(data, **slots):
    country = slots['country']
    k = int(slots['k'])
    top_n = int(slots['top_n'])
    
    try:
        building_ids = data.buildings(country=country)
    except Exception:
        raise ValueError(f"No buildings found in {country}")
        
    if not building_ids:
        raise ValueError("No buildings found for the specified country")

    readings_df = data.readings(building_ids)
    
    valid_cols = [col for col in readings_df.columns if not readings_df[col].isna().all()]
    if not valid_cols:
        raise ValueError("No valid consumption data available for these buildings")
        
    X = readings_df[valid_cols].copy()
    mask = ~X.isna().any(axis=1)
    X_clean = X[mask]
    
    if X_clean.empty:
        raise ValueError("No complete rows available after removing NaNs")

    clusterer = KMeans(n_clusters=k, random_state=0, n_init=10)
    clusterer.fit(X_clean)
    
    labels = clusterer.labels_
    
    building_info = data.metadata()
    info_dict = dict(zip(building_ids, [building_info.loc[bid] for bid in building_ids]))
    
    total_consumption = {}
    for col in valid_cols:
        total_consumption[col] = readings_df[col].sum()
        
    cluster_totals = np.zeros(k)
    for i, col in enumerate(valid_cols):
        mask = labels == i
        if mask.any():
            cluster_totals[i] += X_clean.loc[mask, col].sum()
            
    sorted_clusters = np.argsort(cluster_totals)[::-1][:top_n]
    
    result = {}
    for idx in sorted_clusters:
        total = cluster_totals[idx]
        matching_buildings = [bid for bid, label in zip(building_ids, labels) if label == idx]
        if not matching_buildings:
            continue
            
        types = set()
        for bid in matching_buildings:
            info = info_dict[bid]
            bt = info.get('building_type', 'Unknown')
            st = info.get('building_subtype', '')
            if st:
                full_type = f"{bt} ({st})"
            else:
                full_type = bt
            types.add(full_type)
            
        result[f"Cluster {idx}"] = {
            "total_consumption_kwh": round(total, 6),
            "building_use_types": sorted(list(types))
        }
        
    return {"answer": result, "unit": None}
