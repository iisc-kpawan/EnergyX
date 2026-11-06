"""EnergyX-Max program for template EDA-Q056-T2.

Using K-Means with {k} clusters, group {building_type} buildings in {country} by normalized electricity consumption per usual occupant.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def answer(data, **slots) -> dict:
    k = slots.get('k')
    building_type = slots.get('building_type')
    country = slots.get('country')
    
    # Filter buildings by type and country
    try:
        building_ids = data.buildings(building_type=building_type, country=country)
    except Exception as e:
        raise ValueError(f"No buildings found with type '{building_type}' in country '{country}'.")

    if not building_ids:
        raise ValueError(f"No buildings found matching criteria: type='{building_type}', country='{country}'.")

    # Get metadata for these buildings to get floor area and estimated occupants
    metadata = data.metadata()
    
    # Ensure we only use the subset of metadata relevant to our building_ids
    # The index is building_id, so we can directly select
    try:
        site_ids = metadata.loc[building_ids, 'site_id']
        floor_areas = metadata.loc[building_ids, 'floor_area_m2']
        
        # Check for estimated_occupants; if missing, estimate based on number_of_floors
        try:
            occupants = metadata.loc[building_ids, 'estimated_occupants']
        except KeyError:
            floors = metadata.loc[building_ids, 'number_of_floors']
            # Assume 4 people per floor as a rough estimate for "usual occupant" density
            occupants = floors * 4.0
            
    except Exception as e:
        raise ValueError(f"Failed to retrieve metadata columns for buildings: {e}")

    # Get energy readings for all selected buildings
    try:
        readings_df = data.readings(building_ids)
    except Exception as e:
        raise ValueError(f"Failed to read energy data for buildings: {e}")

    if readings_df.empty or len(readings_df.columns) == 0:
        raise ValueError("No energy consumption data found for the specified buildings.")

    # Sum total kWh per building from the available valid readings
    # readings_df has columns as building_ids, index as datetime. We sum over time (axis=0).
    total_kwh = readings_df.sum(axis=0)  # This returns a Series indexed by building_id
    
    # Ensure alignment between total_kwh and other metadata series
    # Convert to numpy arrays for consistent indexing if necessary, but pandas handles alignment well.
    # We will reconstruct the dataframe carefully to avoid length mismatches.
    
    df_features = pd.DataFrame({
        'building_id': list(building_ids),
        'total_kwh': total_kwh.values,  # This is aligned with building_ids order if we are careful
        'floor_area_m2': floor_areas.values,
        'occupants': occupants.values
    })

    # Re-verify lengths just in case of any subtle misalignment (though unlikely with pandas)
    if not df_features['total_kwh'].equals(total_kwh):
        # If there's a mismatch, it might be due to NaN handling or index order. 
        # Let's explicitly map values by building_id to be safe.
        total_kwh_series = pd.Series(total_kwh, index=building_ids)
        floor_areas_series = pd.Series(floor_areas, index=building_ids)
        occupants_series = pd.Series(occupants, index=building_ids)
        
        df_features = pd.DataFrame({
            'building_id': building_ids,
            'total_kwh': total_kwh_series.values,
            'floor_area_m2': floor_areas_series.values,
            'occupants': occupants_series.values
        })

    # Calculate consumption intensity: kWh / (floor_area * occupants)
    denominator = df_features['floor_area_m2'] * df_features['occupants']
    
    # Avoid division by zero
    valid_mask = (denominator != 0)
    if not valid_mask.all():
        raise ValueError("Some buildings have zero floor area or zero estimated occupants.")
        
    df_features['consumption_intensity'] = np.where(valid_mask, 
                                                    df_features['total_kwh'] / denominator, 
                                                    np.nan)
    
    # Remove rows with NaN intensity for clustering (though KMeans can handle NaN if we drop them)
    # The prompt implies grouping by this metric, so buildings without valid data shouldn't be clustered.
    df_clustering = df_features.dropna(subset=['consumption_intensity'])
    
    if len(df_clustering) == 0:
        raise ValueError("No buildings with valid consumption intensity data found.")

    # Normalize the consumption intensity to 0-1 range for K-Means (optional but good practice)
    min_val = df_clustering['consumption_intensity'].min()
    max_val = df_clustering['consumption_intensity'].max()
    
    if max_val == min_val:
        raise ValueError("All buildings have identical consumption intensity; clustering not possible.")
        
    normalized_intensity = (df_clustering['consumption_intensity'] - min_val) / (max_val - min_val)
    
    # Perform K-Means
    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(normalized_intensity.values.reshape(-1, 1))
    except ValueError as e:
        if "n_samples" in str(e):
            raise ValueError("Not enough buildings to form the requested number of clusters.")
        raise
    
    # Prepare the record output
    cluster_stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_buildings = df_clustering.loc[mask, 'building_id'].tolist()
        cluster_avg_intensity = df_clustering.loc[mask, 'consumption_intensity'].mean()
        
        cluster_stats[str(label)] = {
            "count": int(mask.sum()),
            "avg_consumption_intensity_kwh_per_m2_per_person": round(float(cluster_avg_intensity), 6),
            "building_ids": sorted(cluster_buildings)
        }
    
    return {"answer": cluster_stats}
