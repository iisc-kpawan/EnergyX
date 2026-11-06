"""Helpers shared by the forecasting tasks Q36 to Q40.

Each question forecasts one named day (or week) from a fixed history that ends just before it.
Model inputs come from the filled series (linear interpolation inside gaps of up to 24 hours) and
errors are measured against real readings only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from energyx_gold.analytics import forecasting

HOURS = forecasting.HOURS
WEEK = forecasting.WEEK
MAX_FILL_HOURS = 24

# Tiny Time Mixers for Q40. get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=168,
# prediction_length=24) selects the 90-30-ft-r2.1 branch: 90 hours of context, 30 outputs, the first
# 24 kept through prediction_filter_length. The commit is pinned so an upstream update can't change answers.
TTM_MODEL = forecasting.TTM_MODEL
TTM_REQUESTED_CONTEXT = 168
TTM_BRANCH = "90-30-ft-r2.1"
TTM_REVISION = "6e5cb8ee51e0634a45637490f5db43148b2fa6be"
TTM_HOURLY_FREQ_TOKEN = forecasting.TTM_HOURLY_FREQ_TOKEN


def hours_of(start: str | pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=periods, freq="h", name="timestamp_local")


def mae(actual: pd.Series, forecast: pd.Series) -> tuple[float, int]:
    """MAE over hours (or days) with both a real value and a forecast, and how many there were."""
    both = pd.concat([actual.rename("y"), forecast.rename("yhat")], axis=1).dropna()
    if both.empty:
        return math.nan, 0
    return float((both["yhat"] - both["y"]).abs().mean()), int(len(both))


def lower(values: dict[str, float]) -> str:
    """The name with the lowest value. A tie goes to the first name in alphabetical order."""
    return min(sorted(values), key=lambda name: values[name])


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def window_evidence(readings: pd.Series, filled: pd.Series, index: pd.DatetimeIndex) -> dict:
    """How many hours of a window have a reading, a filled value, or neither."""
    valid = int(readings.reindex(index).notna().sum())
    usable = int(filled.reindex(index).notna().sum())
    return {"hours": len(index), "valid": valid, "filled": usable - valid, "missing": len(index) - usable}


def autoarima(y: np.ndarray, season_length: int, horizon: int) -> tuple[np.ndarray, dict]:
    """statsforecast AutoARIMA with default settings; the forecast and the chosen model."""
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA

    model = AutoARIMA(season_length=season_length)
    model.fit(np.asarray(y, dtype=float))
    forecast = np.asarray(model.predict(h=horizon)["mean"], dtype=float)
    p, q, P, Q, s, d, D = (int(v) for v in model.model_["arma"])
    chosen = {
        "model": arima_string(model.model_).strip(),
        "order": [p, d, q],
        "seasonal_order": [P, D, Q, s],
        "coefficients": {k: rounded(v) for k, v in model.model_["coef"].items()},
        "aicc": rounded(model.model_["aicc"]),
    }
    return forecast, chosen


_TTM = {}


def ttm_model():
    """The checkpoint get_model selects for context 168 and prediction 24, at the pinned commit."""
    import torch
    from tsfm_public.toolkit.get_model import get_model

    if "model" not in _TTM:
        torch.set_num_threads(1)
        key = get_model(TTM_MODEL, context_length=TTM_REQUESTED_CONTEXT, prediction_length=HOURS,
                        return_model_key=True)
        if key != TTM_BRANCH:
            raise RuntimeError(f"get_model now selects {key!r}, not the pinned {TTM_BRANCH!r}")
        _TTM["model"] = get_model(TTM_MODEL, context_length=TTM_REQUESTED_CONTEXT, prediction_length=HOURS,
                                  model_revision=TTM_REVISION).eval()
    return _TTM["model"]


def ttm_forecast(history: pd.Series, index: pd.DatetimeIndex, seed: int) -> tuple[pd.Series, int]:
    """Zero-shot TTM forecast of `index` from the end of `history`.

    The model reads only its own context length (90 hours for the selected checkpoint), so the last
    90 hours are passed. They are standardised by their own mean and population standard deviation,
    as in docs/ml-protocol.md, and the first 24 outputs are the forecast.
    """
    import torch

    torch.manual_seed(seed)
    model = ttm_model()
    context_hours = int(model.config.context_length)
    window = history.to_numpy(dtype=float)[-context_hours:]
    mean, std = float(window.mean()), float(window.std(ddof=0)) or 1.0
    x = torch.tensor((window - mean) / std, dtype=torch.float32).reshape(1, context_hours, 1)
    with torch.no_grad():
        freq = torch.tensor([TTM_HOURLY_FREQ_TOKEN], dtype=torch.long)
        y = model(past_values=x, freq_token=freq).prediction_outputs[0, :HOURS, 0].numpy()
    return pd.Series(y.astype(float) * std + mean, index=index), context_hours
