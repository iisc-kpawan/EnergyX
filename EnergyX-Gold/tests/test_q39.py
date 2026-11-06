"""Independent check of Q39: AutoARIMA against LightGBM, Peacock_office_Major, 1 March 2016.

Recomputed with pandas over BDG2's electricity_cleaned.csv, calling statsforecast and LightGBM
directly: no snapshot, no DuckDB, no energyx_gold.analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TASK_ID = "Q39"
BUILDING = "Peacock_office_Major"
TARGET = pd.Timestamp("2016-03-01 00:00")
HISTORY = 168
HORIZON = 24


def fill(values: pd.Series, limit: int = 24) -> pd.Series:
    gap = values.isna()
    run = (gap != gap.shift()).cumsum()
    length = gap.groupby(run).transform("sum")
    return values.interpolate(limit_area="inside").where(~(gap & (length > limit)))


@pytest.fixture(scope="module")
def want(raw) -> dict:
    import lightgbm as lgb
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    values = raw.electricity_cleaned[BUILDING].astype(float)
    history_index = pd.date_range(TARGET - pd.Timedelta(hours=HISTORY), periods=HISTORY, freq="h")
    target_index = pd.date_range(TARGET, periods=HORIZON, freq="h")
    history = fill(values.reindex(history_index)).to_numpy()
    assert not np.isnan(history).any()
    actual = values.reindex(target_index).to_numpy()

    frame = pd.DataFrame({"unique_id": BUILDING, "ds": history_index, "y": history})
    arima = StatsForecast(models=[AutoARIMA(season_length=24)], freq="h", n_jobs=1).forecast(
        df=frame, h=HORIZON)["AutoARIMA"].to_numpy()

    # Rows 24..167 of the history have a lag-24 value inside it.
    train_index = history_index[24:]
    X_train = np.column_stack([train_index.hour, train_index.dayofweek, history[:-24]])
    y_train = history[24:]
    X_test = np.column_stack([target_index.hour, target_index.dayofweek, history[-24:]])
    booster = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                random_state=0, deterministic=True, n_jobs=1, verbose=-1)
    booster.fit(pd.DataFrame(X_train, columns=["hour", "day_of_week", "lag_24"]), y_train)
    lgbm = booster.predict(pd.DataFrame(X_test, columns=["hour", "day_of_week", "lag_24"]))

    ok = ~np.isnan(actual)
    return {"autoarima_mae": float(np.abs(arima - actual)[ok].mean()),
            "lightgbm_mae": float(np.abs(lgbm - actual)[ok].mean())}


def test_maes_match(gold, want):
    answer = gold(TASK_ID)["result"]["answer"]
    for key, value in want.items():
        assert answer[key] == pytest.approx(value, rel=1e-6, abs=5e-7)


def test_winner_matches(gold, want):
    better = "AutoARIMA" if want["autoarima_mae"] < want["lightgbm_mae"] else "LightGBM"
    assert gold(TASK_ID)["result"]["answer"]["lower_mae_model"] == better
