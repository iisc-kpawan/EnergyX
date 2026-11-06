"""EnergyX-Max program for template CS-302-T1.

Using a seasonal naive forecast with a {horizon_hours}-hour horizon, what is the predicted electricity demand for each of these five buildings?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
import numpy as np

def answer(data, **slots):
    horizon = slots.get("horizon_hours", 24)
    buildings = ["B0001", "B0002", "B0003", "B0004", "B0005"]
    result = {}
    for b_id in buildings:
        try:
            readings = data.readings([b_id])
            if b_id not in readings.columns or readings[b_id].isna().all():
                raise ValueError(f"No valid data for {b_id}")
            last_idx = readings.index[-1]
            forecast_val = readings[b_id].iloc[last_idx - horizon]
            result[b_id] = float(np.round(forecast_val, 6)) if pd.notna(forecast_val) else None
        except Exception:
            result[b_id] = None
    return {"answer": result, "unit": "kWh"}
