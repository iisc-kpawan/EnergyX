"""EnergyX-Max program for template CS-370-T1.

Which building-use type is most similar to {building_id} based on clustering its hourly electricity consumption?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans

def answer(data, **slots):
    building_id = slots['building_id']
    try:
        readings_df = data.readings([building_id])
        if building_id not in readings_df.columns:
            raise ValueError(f"Building {building_id} not found")
        col = readings_df[building_id]
        if col.isna().all():
            raise ValueError("No valid consumption data for the building")
        mask = ~col.isna()
        if mask.sum() < 8760 * 0.9:
            raise ValueError("Insufficient valid hours (needs >= 90%)")
        values = col[mask].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(values)
        cluster_0_count = np.sum(labels == 0)
        cluster_1_count = np.sum(labels == 1)
        if cluster_0_count > cluster_1_count:
            dominant_label = 0
        else:
            dominant_label = 1
        other_label = 1 - dominant_label
        all_building_ids = data.buildings()
        other_buildings = [b_id for b_id in all_building_ids if b_id != building_id]
        if not other_buildings:
            raise ValueError("No other buildings to compare")
        other_consumptions = {}
        for b_id in other_buildings:
            try:
                r = data.readings([b_id])
                if b_id not in r.columns: continue
                c = r[b_id]
                if c.isna().all(): continue
                m = ~c.isna()
                if m.sum() < 8760 * 0.9: continue
                other_consumptions[b_id] = c[m].values.reshape(-1, 1)
            except Exception:
                continue
        if not other_consumptions:
            raise ValueError("No other buildings with sufficient data to compare")
        best_match = None
        best_dist = float('inf')
        for b_id, vals in other_consumptions.items():
            kmeans_other = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels_other = kmeans_other.fit_predict(vals)
            l0_c = np.sum(labels_other == 0)
            l1_c = np.sum(labels_other == 1)
            if l0_c > l1_c:
                dom_l = 0
                other_l = 1
            else:
                dom_l = 1
                other_l = 0
            dist = np.mean((labels[dom_l] != labels_other[other_l]).astype(float))
            if dist < best_dist:
                best_dist = dist
                best_match = b_id
        if best_match is None:
            raise ValueError("No suitable match found")
        metadata = data.metadata()
        type_map = {b_id: v['building_type'] for b_id, v in metadata.iterrows()}
        building_type = type_map.get(best_match)
        if building_type is None:
            raise ValueError(f"Building type not found for {best_match}")
        return {"answer": building_type, "unit": None}
    except Exception as e:
        raise ValueError(str(e))
