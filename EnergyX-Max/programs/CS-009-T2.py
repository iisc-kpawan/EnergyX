"""EnergyX-Max program for template CS-009-T2.

Using K-Means with {k} clusters, identify groups of buildings at the {site_id} site that share similar data availability patterns.

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def answer(data, **slots) -> dict:
    k = slots.get('k')
    site_id = slots.get('site_id')
    
    # Get building IDs for the specified site
    building_ids = data.buildings(site_id=site_id)
    
    if not building_ids:
        raise ValueError(f"No buildings found at site '{site_id}'.")
    
    # Define a reasonable date range to capture monthly patterns.
    # We will use the full available history for these buildings to define their pattern.
    try:
        # Get the most recent reading to determine the end date
        all_readings = data.readings(building_ids=building_ids[:1])
        if len(all_readings) == 0:
             raise ValueError("No readings found for buildings at site.")
        
        end_date = all_readings.index.max() + pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=730) # Approx 2 years
        
    except Exception:
        raise ValueError("Could not determine date range for analysis.")

    # Get readings for all buildings in the site
    try:
        df_readings = data.readings(building_ids=building_ids, start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Error retrieving readings: {str(e)}")
        
    if df_readings.empty:
        raise ValueError("No valid readings found within the analysis period.")

    # Create a binary mask for each building: 1 if any hour in that month is not NaN, else 0
    # We group by month and check if sum(valid) > 0
    
    # Get all unique months present in the readings
    min_year = df_readings.index.year.min()
    max_year = df_readings.index.year.max()
    
    # Create a dataframe where index is building_id and columns are (year, month). Value: 1 if present.
    feature_matrix = np.zeros((len(building_ids), len(building_ids) * 12), dtype=int) # Placeholder shape
    
    # We need to map each building's data to a specific set of month columns.
    # Let's construct the matrix row by row for each building.
    
    current_idx = 0
    for idx, building_id in enumerate(building_ids):
        col_data = df_readings[building_id]
        if col_data.empty:
            continue
            
        # For each month within the range of this building's data, check presence
        # We will create a row for this building with 1s where data exists.
        # To make KMeans work, we need a fixed number of columns representing time periods.
        # Let's define the time periods as all months from min_year to max_year across ALL buildings.
        
        # Re-calculate feature matrix dimensions based on global date range
        pass

    # Correct approach: Define global month columns first
    month_columns = pd.MultiIndex.from_product([[min_year, max_year], range(1, 13)], names=['year', 'month'])
    
    # Initialize matrix with correct shape
    n_months = len(month_columns)
    feature_matrix = np.zeros((len(building_ids), n_months), dtype=int)
    
    for idx, building_id in enumerate(building_ids):
        col_data = df_readings[building_id]
        if col_data.empty:
            continue
            
        # For each month column, check if there is at least one valid reading
        for i, (year, month) in enumerate(month_columns):
            mask = (col_data.index.year == year) & (col_data.index.month == month)
            if mask.any():
                feature_matrix[idx, i] = 1
    
    if feature_matrix.sum() == 0:
        raise ValueError("No data availability patterns found to cluster.")

    # Handle case where k > number of unique patterns or similar issues
    try:
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(feature_matrix)
    except ValueError as e:
        if "n_samples" in str(e):
            raise ValueError(f"Not enough data points to form {k} clusters.")
        raise

    # Prepare the record output
    result = {}
    for i in range(k):
        cluster_buildings = [building_ids[j] for j, label in enumerate(labels) if label == i]
        if cluster_buildings:
            # Sort building IDs for consistency
            result[str(i)] = sorted(cluster_buildings)
    
    return {"answer": result, "unit": None}
