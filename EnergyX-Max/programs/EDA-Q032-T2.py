"""EnergyX-Max program for template EDA-Q032-T2.

Using K-Means with {k} clusters on hourly consumption, which cluster exhibits the highest average demand variability (standard deviation) when comparing {building_type} buildings against {building_type_2} buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import numpy as np
    from sklearn.cluster import KMeans
    
    k = slots['k']
    bt1 = slots['building_type']
    bt2 = slots['building_type_2']
    
    # Retrieve building IDs for the two specified building types
    try:
        ids1 = data.buildings(building_type=bt1)
        ids2 = data.buildings(building_type=bt2)
    except Exception as e:
        raise ValueError(f"Error retrieving buildings: {str(e)}")
    
    if not ids1 or not ids2:
        raise ValueError("No buildings found for the specified types.")
        
    # Combine IDs and sort to ensure consistent ordering
    all_ids = sorted(list(set(ids1 + ids2)))
    
    if len(all_ids) < k:
        raise ValueError(f"Not enough unique buildings ({len(all_ids)}) for {k} clusters.")

    # Retrieve consumption data for the combined set of buildings
    try:
        df_readings = data.readings(all_ids)
    except Exception as e:
        raise ValueError(f"Failed to retrieve readings: {str(e)}")
        
    if df_readings.empty:
        raise ValueError("No consumption data available.")

    # Calculate standard deviation for each building based on valid (non-NaN) readings
    std_devs = {}
    for bid in all_ids:
        col = df_readings[bid]
        vals = col.dropna()
        if len(vals) > 0:
            std_devs[bid] = float(vals.std(ddof=1))
        else:
            std_devs[bid] = 0.0
            
    # Filter out buildings with no data (std dev 0 and no readings, though spec says only metered)
    # We keep buildings with at least one reading. If all are NaN, std is 0.
    active_buildings = [bid for bid, std in std_devs.items() if not np.isnan(std)]
    
    if len(active_buildings) < k:
        raise ValueError(f"Not enough buildings with data ({len(active_buildings)}) for {k} clusters.")
        
    # Prepare data for KMeans
    X = np.array([std_devs[b] for b in active_buildings]).reshape(-1, 1)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    
    # Create a mapping from building_id to its type
    type_map = {}
    for bid in ids1:
        type_map[bid] = bt1
    for bid in ids2:
        type_map[bid] = bt2
        
    # Calculate average std dev for each cluster
    cluster_stats = {}
    for i, label in enumerate(labels):
        if label not in cluster_stats:
            cluster_stats[label] = []
        cluster_stats[label].append(std_devs[active_buildings[i]])
        
    avg_std_devs = {label: np.mean(stats) for label, stats in cluster_stats.items()}
    
    # Find the cluster with the highest average demand variability
    max_cluster = max(avg_std_devs.keys(), key=lambda x: avg_std_devs[x])
    max_variability = avg_std_devs[max_cluster]
    
    return {
        "answer": max_cluster,
        "unit": None
    }
