"""EnergyX-Max program for template CS-346-T1.

Is the hourly consumption anomaly for {building_id} on {date} still statistically significant after adjusting for the outdoor air temperature at the time?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

def answer(data, **slots):
    building_id = slots['building_id']
    date_str = slots['date']
    try:
        start_ts, end_ts = data.year_range(2017) # Assuming year 2017 based on example date; adjust if needed dynamically or assume single year context.
        # Re-evaluating year assumption: The question asks about a specific date. We need readings around that date to fit the model.
        # Let's define a window around the date to get enough data points for regression, or use the whole available history if possible.
        # However, standard practice for such questions usually implies using available data up to that point or a fixed window.
        # Given the constraints and lack of 'year' slot, we must infer or use a reasonable window.
        # Let's try to get readings for the year containing the date.
        year = int(date_str.split('-')[0])
        start_ts, end_ts = data.year_range(year)
        
        readings_df = data.readings([building_id], start=start_ts, end=end_ts)
        if building_id not in readings_df.columns:
            raise ValueError(f"Building {building_id} not found.")
            
        # Get weather data for the same period
        # We need to know the site_id. Let's try to get it from metadata or assume we can query weather by building_id?
        # The API says data.weather(site_id, ...). We need site_id.
        # Let's check if we can get site_id from metadata.
        meta = data.metadata()
        if building_id in meta.index:
            site_id = meta.loc[building_id, 'site_id']
        else:
             raise ValueError(f"Cannot find site_id for {building_id}.")
             
        weather_df = data.weather(site_id, start=start_ts, end=end_ts)
        
        # Prepare data
        date_idx = pd.to_datetime(readings_df.index)
        readings_series = readings_df[building_id]
        temp_series = weather_df['air_temperature']
        
        # Ensure alignment and drop NaNs
        combined = pd.concat([readings_series, temp_series], axis=1).dropna()
        
        if len(combined) < 2:
            raise ValueError("Insufficient data points for regression.")
            
        X = combined[['air_temperature']].values
        y = combined[building_id].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get prediction and residual for the specific date
        target_date = pd.to_datetime(date_str)
        if target_date not in date_idx:
            raise ValueError(f"Date {date_str} not found in data.")
            
        pred_temp = temp_series.loc[target_date]
        actual_consumption = readings_series.loc[target_date]
        
        # Predict consumption based on temperature
        predicted_consumption = model.predict([[pred_temp]])[0]
        residual = actual_consumption - predicted_consumption
        
        # Determine statistical significance. 
        # Without explicit p-value calculation (requires statsmodels or manual t-test), we can check if residual is large relative to noise.
        # However, the prompt asks "Is it still statistically significant?".
        # A simple heuristic: Is the residual significantly different from 0?
        # We can estimate standard error of the regression or use a simple threshold if not provided.
        # Given the constraints (60 lines), let's assume we check if the residual is non-zero and perhaps large relative to typical variation.
        # But "statistically significant" usually implies a p-value < 0.05.
        # Let's try to compute a simple t-statistic: residual / std_residual_of_that_point? 
        # Or simply, if the model explains variance well, is the specific point an outlier?
        # Alternative interpretation: The question might imply checking if the consumption deviates from the expected value given temp.
        # Let's assume we need to check if the residual is significant compared to the standard deviation of residuals.
        
        y_pred = model.predict(combined[['air_temperature']])
        residuals = combined[building_id] - y_pred
        std_residual = residuals.std()
        
        # Simple t-statistic approximation for this point (assuming normality of errors)
        # t = residual / (std_residual / sqrt(n)) is not quite right for a single point without degrees of freedom adjustment.
        # Let's use a simpler approach: Is |residual| > 2 * std_residual? This is a rough proxy for significance at ~95% confidence if errors are homoscedastic.
        # However, the most robust way with sklearn is hard without statsmodels.
        # Let's assume the question expects a binary answer based on whether the residual is "large".
        # Or perhaps the question implies: "After adjusting for temp, is there still an anomaly?" -> Is residual != 0 significantly?
        # Let's define significance as |residual| > 2 * std_residual (approx 95% CI).
        
        is_significant = abs(residual) > 2.0 * std_residual
        
        if is_significant:
            return {"answer": "Yes", "unit": None}
        else:
            return {"answer": "No", "unit": None}
            
    except Exception as e:
        raise ValueError(str(e))
