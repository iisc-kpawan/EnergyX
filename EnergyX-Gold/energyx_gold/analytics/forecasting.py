"""Day-ahead forecasting for gold tasks: the models the questions name, and error metrics.

The protocol (docs/ml-protocol.md):
- Each forecast is made at 00:00 site-local time for the 24 hours of that day.
- A period named in a question is split by whole days: the first 80% train, the last 20% test.
- Inputs come from `fill_gaps` (linear, gaps up to 24 hours). Errors are measured only against
  hours with a real reading.
- Every model is deterministic: seeds come from `context.seed`, and each runs on one thread.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOURS = 24
WEEK = 168
TRAIN_SHARE = 0.8

LIGHTGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)
KNN_NEIGHBOURS = 5
ARIMA_ORDER = (2, 0, 1)
ARIMA_SEASONAL_ORDER = (1, 0, 1, 24)
TTM_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
TTM_CONTEXT = 512
# The 512-48-ft-r2.1 branch at this commit. Pinned so an upstream update can't change answers.
TTM_REVISION = "b972f0c22190b7502764526004d16e2b4ed39e8c"
TTM_HOURLY_FREQ_TOKEN = 7  # tsfm_public DEFAULT_FREQUENCY_MAPPING["h"]


# -- splits and metrics --------------------------------------------------------


def split_days(first_day: str | pd.Timestamp, last_day: str | pd.Timestamp,
               train_share: float = TRAIN_SHARE) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Whole days from `first_day` to `last_day` inclusive: the first floor(80%) train, the rest test."""
    days = list(pd.date_range(pd.Timestamp(first_day).normalize(), pd.Timestamp(last_day).normalize(), freq="D"))
    n_train = math.floor(len(days) * train_share)
    return days[:n_train], days[n_train:]


def day_hours(days: list[pd.Timestamp]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(HOURS)], name="timestamp_local")


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """MAE, MSE, RMSE (kWh units) and MAPE (%), over hours with a real reading and a prediction.

    MAPE leaves out hours whose reading is 0.
    """
    both = pd.concat([actual.rename("y"), predicted.rename("yhat")], axis=1).dropna()
    error = both["yhat"] - both["y"]
    nonzero = both["y"] != 0
    mse = float((error ** 2).mean()) if len(both) else math.nan
    return {
        "MAE": float(error.abs().mean()) if len(both) else math.nan,
        "MSE": mse,
        "RMSE": math.sqrt(mse) if len(both) else math.nan,
        "MAPE": float((error[nonzero].abs() / both.loc[nonzero, "y"].abs()).mean() * 100) if nonzero.any() else math.nan,
        "hours": int(len(both)),
    }


def rounded(values: dict, digits: int = 6) -> dict:
    return {k: (round(v, digits) if isinstance(v, float) and not math.isnan(v) else v) for k, v in values.items()}


# -- models --------------------------------------------------------------------


def seasonal_naive(filled: pd.Series, test_index: pd.DatetimeIndex, season: int = WEEK) -> pd.Series:
    """The value one season earlier: y(t - 168 hours) for a weekly season."""
    return filled.reindex(test_index - pd.Timedelta(hours=season)).set_axis(test_index)


def lag_features(filled: pd.Series, index: pd.DatetimeIndex, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calendar features and lags of at least 24 hours, so a day-ahead forecast never needs a prediction as input."""
    frame = pd.DataFrame(index=index)
    frame["hour"] = index.hour
    frame["day_of_week"] = index.dayofweek
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    for lag in (24, 48, WEEK):
        frame[f"lag_{lag}"] = filled.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    if exog is not None:
        frame = frame.join(exog.reindex(index))
    return frame


def lightgbm(filled: pd.Series, actual: pd.Series, train_index: pd.DatetimeIndex, test_index: pd.DatetimeIndex,
             seed: int, exog: pd.DataFrame | None = None) -> pd.Series:
    """LightGBM on calendar features, lags 24/48/168 and optional exogenous columns. Trained on real readings only."""
    from lightgbm import LGBMRegressor

    X_train = lag_features(filled, train_index, exog)
    y_train = actual.reindex(train_index)
    keep = y_train.notna()
    if not keep.any():  # no training reading: no forecast
        return pd.Series(np.nan, index=test_index)
    model = LGBMRegressor(random_state=seed, **LIGHTGBM_PARAMS)
    model.fit(X_train[keep], y_train[keep])
    return pd.Series(model.predict(lag_features(filled, test_index, exog)), index=test_index)


def _windows(filled: pd.Series, days: list[pd.Timestamp], exog: pd.DataFrame | None):
    X, Y, kept = [], [], []
    for day in days:
        history = filled.reindex(pd.date_range(day - pd.Timedelta(hours=WEEK), periods=WEEK, freq="h"))
        target = filled.reindex(pd.date_range(day, periods=HOURS, freq="h"))
        row = history.to_numpy()
        if exog is not None:
            row = np.concatenate([row, exog.reindex(target.index).to_numpy().ravel()])
        X.append(row)
        Y.append(target.to_numpy())
        kept.append(day)
    return np.array(X, dtype=float), np.array(Y, dtype=float), kept


def knn(filled: pd.Series, train_days: list[pd.Timestamp], test_days: list[pd.Timestamp],
        exog: pd.DataFrame | None = None, neighbours: int = KNN_NEIGHBOURS) -> pd.Series:
    """KNN on one-week windows: the 168 hours before midnight predict the next 24.

    Windows start every day, so consecutive windows overlap by 144 hours. Windows use the filled series
    (gaps up to 24 hours interpolated), for targets as well as inputs. Inputs are min-max scaled with
    the training windows' range. Training windows with any NaN are skipped; a test window with NaN
    gets no forecast, and so does every test day when no training window is complete.
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import MinMaxScaler

    X, Y, _ = _windows(filled, train_days, exog)
    complete = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X, Y = X[complete], Y[complete]
    X_test, _, days = _windows(filled, test_days, exog)
    if len(X) == 0:  # no complete training window: no forecast
        return pd.Series(np.nan, index=day_hours(days))
    scaler = MinMaxScaler().fit(X)
    model = KNeighborsRegressor(n_neighbors=min(neighbours, len(X))).fit(scaler.transform(X), Y)
    predictions = np.full((len(days), HOURS), np.nan)
    ok = ~np.isnan(X_test).any(axis=1)
    if ok.any():
        predictions[ok] = model.predict(scaler.transform(X_test[ok]))
    return pd.Series(predictions.ravel(), index=day_hours(days))


def arima(filled: pd.Series, train_index: pd.DatetimeIndex, test_days: list[pd.Timestamp]) -> pd.Series:
    """Seasonal ARIMA (2,0,1)x(1,0,1,24) with a constant, fitted once on the training hours.

    Each test day is forecast 24 steps ahead from the readings up to its midnight, with the fitted
    parameters kept fixed.
    """
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = filled.reindex(train_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(train.to_numpy(), order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL_ORDER, trend="c",
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        full_index = pd.date_range(train_index[0], test_days[-1] + pd.Timedelta(hours=HOURS - 1), freq="h")
        extended = fitted.apply(filled.reindex(full_index).to_numpy())
        out = []
        for day in test_days:
            start = full_index.get_loc(day)
            forecast = extended.get_prediction(start=start, end=start + HOURS - 1, dynamic=0).predicted_mean
            out.append(pd.Series(forecast, index=pd.date_range(day, periods=HOURS, freq="h")))
    return pd.concat(out)


_TTM = {}


def ttm(filled: pd.Series, test_days: list[pd.Timestamp], seed: int, context_length: int = TTM_CONTEXT) -> pd.Series:
    """Tiny Time Mixers (granite-timeseries-ttm-r2), zero-shot, on CPU.

    For each test day the 512 hours before midnight, standardised by their own mean and standard
    deviation, predict the day's 24 hours. A context with NaN gets no forecast.
    """
    import torch
    from tsfm_public.toolkit.get_model import get_model

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if context_length not in _TTM:
        _TTM[context_length] = get_model(TTM_MODEL, model_revision=TTM_REVISION, context_length=context_length, prediction_length=HOURS).eval()
    model = _TTM[context_length]
    out = []
    for day in test_days:
        window = filled.reindex(pd.date_range(day - pd.Timedelta(hours=context_length), periods=context_length, freq="h"))
        index = pd.date_range(day, periods=HOURS, freq="h")
        if window.isna().any():
            out.append(pd.Series(np.nan, index=index))
            continue
        mean, std = float(window.mean()), float(window.std(ddof=0)) or 1.0
        x = torch.tensor(((window.to_numpy() - mean) / std), dtype=torch.float32).reshape(1, context_length, 1)
        with torch.no_grad():
            freq = torch.tensor([TTM_HOURLY_FREQ_TOKEN], dtype=torch.long)
            y = model(past_values=x, freq_token=freq).prediction_outputs[0, :HOURS, 0].numpy()
        out.append(pd.Series(y * std + mean, index=index))
    return pd.concat(out)
