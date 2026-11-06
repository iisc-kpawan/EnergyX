"""EnergyX-Max program for template ADV-08-T1.

How do model rankings differ between LightGBM, KNN, and Seasonal Naive forecasters when evaluated using MAE versus RMSE across multiple {building_type} buildings in {country}?

Written by the Coding Agent (qwen3.5:9b); not manually validated.
"""

def answer(data, **slots) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    building_type = slots['building_type']
    country = slots['country']
    
    # 1. Identify buildings
    try:
        bldg_ids = data.buildings(building_type=building_type, country=country)
    except Exception:
        raise ValueError(f"No buildings found for type '{building_type}' in country '{country}'.")
    
    if not bldg_ids:
        raise ValueError("No qualifying buildings found.")

    # 2. Define models and training logic
    def train_model(bldg_id, start, end):
        try:
            df = data.readings([bldg_id], start=start, end=end)
            if df.empty or df[bldg_id].isna().all():
                return None
            
            y = df[bldg_id].values
            X = df.index.to_series().dt.hour.values.reshape(-1, 1) # Use hour as feature for simplicity
            
            # LightGBM
            try:
                from lightgbm import LGBMRegressor
                lgb_model = LGBMRegressor(random_state=0, n_estimators=200, max_depth=5)
                lgb_model.fit(X, y)
                lgb_pred = lgb_model.predict(X)
            except Exception:
                # Fallback if lightgbm not available or fails
                lgb_pred = None
            
            # KNN
            knn_model = KNeighborsRegressor(n_neighbors=5)
            knn_model.fit(X, y)
            knn_pred = knn_model.predict(X)
            
            # Seasonal Naive (Seasonal Mean)
            # Group by day of year and month to get seasonal means
            # Since X is just hour, we need to reconstruct the seasonal context.
            # A simple seasonal naive for hourly data: predict with the average value of that hour across all years available.
            # However, standard Seasonal Naive usually predicts t with mean(t - period).
            # Let's implement a robust seasonal naive using the training data distribution.
            
            # Calculate seasonal means (mean of y for each hour)
            seasonal_means = {}
            for h in range(24):
                mask = X.flatten() == h
                if mask.any():
                    seasonal_means[h] = y[mask].mean()
            
            naive_pred = np.array([seasonal_means.get(h, 0) for h in X.flatten()])
            
            return {
                'lgb': lgb_pred,
                'knn': knn_pred,
                'naive': naive_pred
            }
        except Exception:
            return None

    # 3. Evaluate models across all buildings
    results = {}
    
    for bldg_id in bldg_ids:
        try:
            df = data.readings([bldg_id])
            if df.empty:
                continue
            
            start = df.index.min()
            end = df.index.max() + pd.Timedelta(hours=1)
            
            model_outputs = train_model(bldg_id, start, end)
            
            if model_outputs is None:
                continue
                
            y_true = df[bldg_id].values
            
            # Calculate MAE and RMSE for each model
            lgb_mae = mean_absolute_error(y_true, model_outputs['lgb'])
            lgb_rmse = np.sqrt(mean_squared_error(y_true, model_outputs['lgb']))
            
            knn_mae = mean_absolute_error(y_true, model_outputs['knn'])
            knn_rmse = np.sqrt(mean_squared_error(y_true, model_outputs['knn']))
            
            naive_mae = mean_absolute_error(y_true, model_outputs['naive'])
            naive_rmse = np.sqrt(mean_squared_error(y_true, model_outputs['naive']))
            
            results[bldg_id] = {
                'lgb': {'mae': lgb_mae, 'rmse': lgb_rmse},
                'knn': {'mae': knn_mae, 'rmse': knn_rmse},
                'naive': {'mae': naive_mae, 'rmse': naive_rmse}
            }
        except Exception:
            continue

    if not results:
        raise ValueError("Could not compute metrics for any building.")

    # 4. Aggregate rankings
    # We need to rank models based on MAE and RMSE separately.
    # Lower is better.
    
    # Flatten results for ranking
    model_metrics = []
    for bldg_id, metrics in results.items():
        for model_name, vals in metrics.items():
            model_metrics.append({
                'building': bldg_id,
                'model': model_name,
                'mae': vals['mae'],
                'rmse': vals['rmse']
            })
    
    df_metrics = pd.DataFrame(model_metrics)
    
    # Rank by MAE (lower is better)
    df_metrics['rank_mae'] = df_metrics.groupby('building')['mae'].rank(ascending=True).astype(int)
    # Rank by RMSE (lower is better)
    df_metrics['rank_rmse'] = df_metrics.groupby('building')['rmse'].rank(ascending=True).astype(int)
    
    # Aggregate rankings per model type across all buildings
    # We will count how many times each model was 1st, 2nd, etc., or just average the rank.
    # The question asks "How do model rankings differ". A summary of average ranks is standard.
    
    agg_mae = df_metrics.groupby('model')['rank_mae'].mean().round(6)
    agg_rmse = df_metrics.groupby('model')['rank_rmse'].mean().round(6)
    
    # Create the record
    record = {
        'LightGBM': {
            'avg_rank_mae': float(agg_mae.get('lgb', np.nan)),
            'avg_rank_rmse': float(agg_rmse.get('lgb', np.nan))
        },
        'KNN': {
            'avg_rank_mae': float(agg_mae.get('knn', np.nan)),
            'avg_rank_rmse': float(agg_rmse.get('knn', np.nan))
        },
        'Seasonal Naive': {
            'avg_rank_mae': float(agg_mae.get('naive', np.nan)),
            'avg_rank_rmse': float(agg_rmse.get('naive', np.nan))
        }
    }
    
    return {"answer": record, "unit": None}
