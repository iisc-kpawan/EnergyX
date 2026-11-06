"""EnergyX-Max program for template ADV-10-T1.

Using an Isolation Forest model, what is the count of electricity-consumption anomalies for {building_type} buildings at the {site_id} site between {date} and {date}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def answer(data, **slots) -> dict:
    building_type = slots['building_type']
    site_id = slots['site_id']
    date_str = slots['date']
    
    # Parse the single date provided in the slot to determine the range.
    # The question asks for anomalies "between {date} and {date}". 
    # This implies a single-day analysis or a range where start=end.
    # We will interpret this as the specific day provided.
    try:
        start_date = pd.to_datetime(date_str)
        end_date = start_date + pd.Timedelta(days=1)
    except Exception:
        raise ValueError(f"Invalid date format: {date_str}")

    # Retrieve building IDs for the specified site and building type
    try:
        building_ids = data.buildings(site_id=site_id, building_type=building_type)
    except Exception:
        raise ValueError(f"No buildings found for site '{site_id}' and type '{building_type}'.")

    if not building_ids:
        raise ValueError("No qualifying buildings found.")

    # Read energy consumption data for the specified range
    try:
        readings = data.readings(building_ids, start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Failed to read energy data: {str(e)}")

    # Filter out buildings with no valid readings in the period
    valid_cols = [col for col in readings.columns if not readings[col].isna().all()]
    
    if len(valid_cols) == 0:
        return {"answer": 0, "unit": None}

    # Prepare data for Isolation Forest
    # We need to handle potential non-numeric columns if any (though kWh should be numeric)
    X = readings[valid_cols].fillna(0).values
    
    # Train the Isolation Forest model
    # Using default parameters as per standard sklearn usage for anomaly detection
    iso_forest = IsolationForest(random_state=0, contamination='auto')
    
    try:
        iso_forest.fit(X)
    except Exception as e:
        raise ValueError(f"Model fitting failed: {str(e)}")

    # Predict anomalies (-1 for outlier, 1 for inlier)
    predictions = iso_forest.predict(X)
    
    # Count the number of anomalies (where prediction is -1)
    anomaly_count = np.sum(predictions == -1)

    return {"answer": int(anomaly_count), "unit": None}
