"""Q39: AutoARIMA against LightGBM for Peacock_office_Major on 1 March 2016.

Question: Compare AutoARIMA and LightGBM for forecasting the next 24 hours of electricity
consumption for `Peacock_office_Major`, starting at 00:00 on 1 March 2016. Use the previous 7 days of
hourly data, and report which model has the lower MAE.

Pinned definition: Peacock_office_Major. History is the 168 hours from 23 February 2016 00:00 to 29
February 23:00, with gaps of up to 24 hours filled linearly. Both models forecast the 24 hours of 1
March 2016. AutoARIMA is statsforecast's AutoARIMA with season_length 24. LightGBM is trained on the
history hours that have a reading 24 hours earlier, with features hour of day, day of week and the
lag-24 reading, and parameters n_estimators 200, learning_rate 0.05, num_leaves 15,
min_child_samples 5, random_state 0, deterministic true; the target day's lag-24 values come from 29
February. The MAE covers the hours of 1 March with a valid reading.

- Both models read the filled history. LightGBM's training rows are 24 February 00:00 to 29
  February 23:00 (144 hours), whose lag-24 value lies inside the history. Targets and lags are the
  filled values; one target hour (26 February 09:00) is filled.
- LightGBM also runs with n_jobs 1 and verbose -1, which don't change the model.

    python -m energyx_gold.tasks.category_08_forecasting.Q39
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_08_forecasting import _helpers as helpers

TASK_ID = "Q39"
BUILDING = "Peacock_office_Major"
TARGET_START = pd.Timestamp("2016-03-01 00:00")
HISTORY_HOURS = helpers.WEEK
HORIZON_HOURS = helpers.HOURS
ARIMA_SEASON_LENGTH = 24
LAG_HOURS = 24
MAX_FILL_HOURS = helpers.MAX_FILL_HOURS
FEATURES = ["hour", "day_of_week", "lag_24"]
LIGHTGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                       random_state=0, deterministic=True, n_jobs=1, verbose=-1)
NAMES = {"autoarima": "AutoARIMA", "lightgbm": "LightGBM"}


def features(values: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "hour": index.hour,
        "day_of_week": index.dayofweek,
        "lag_24": values.reindex(index - pd.Timedelta(hours=LAG_HOURS)).to_numpy(),
    }, index=index)


def main(context: GoldContext | None = None) -> dict:
    from lightgbm import LGBMRegressor

    context = context or GoldContext()
    history_index = helpers.hours_of(TARGET_START - pd.Timedelta(hours=HISTORY_HOURS), HISTORY_HOURS)
    target_index = helpers.hours_of(TARGET_START, HORIZON_HOURS)
    readings = series.hourly(context, [BUILDING], history_index[0], target_index[-1] + pd.Timedelta(hours=1))[BUILDING]
    filled = series.fill_gaps(readings.reindex(history_index), limit=MAX_FILL_HOURS)
    if filled.isna().any():
        raise ValueError(f"{BUILDING}: history has a gap longer than {MAX_FILL_HOURS} hours")
    actual = readings.reindex(target_index)

    arima_values, chosen = helpers.autoarima(filled.to_numpy(), ARIMA_SEASON_LENGTH, HORIZON_HOURS)

    train = features(filled, history_index).assign(y=filled.to_numpy()).dropna()
    model = LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(train[FEATURES], train["y"])
    X_target = features(filled, target_index)
    forecasts = {
        "autoarima": pd.Series(arima_values, index=target_index),
        "lightgbm": pd.Series(model.predict(X_target[FEATURES]), index=target_index),
    }
    scores = {name: helpers.mae(actual, forecast) for name, forecast in forecasts.items()}

    answer = {f"{name}_mae": helpers.rounded(scores[name][0]) for name in NAMES}
    answer["lower_mae_model"] = NAMES[helpers.lower({name: answer[f"{name}_mae"] for name in NAMES})]
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": {f"{name}_mae": "kWh" for name in NAMES}},
        "evidence": {
            "definition": {
                "building": BUILDING,
                "history": f"{history_index[0]} to {history_index[-1]} ({HISTORY_HOURS} hours), site-local",
                "target": f"{target_index[0]} to {target_index[-1]}, site-local",
                "max_fill_hours": MAX_FILL_HOURS,
                "autoarima": {"library": "statsforecast", "season_length": ARIMA_SEASON_LENGTH,
                              "other_settings": "defaults (stepwise search)"},
                "lightgbm": {"features": FEATURES, **LIGHTGBM_PARAMS,
                             "training_rows": "history hours whose lag-24 hour is in the history; filled values",
                             "target_lag_24": "29 February, filled history"},
                "scoring": "hours of 1 March with a valid reading",
            },
            "history_window": helpers.window_evidence(readings, filled, history_index),
            "filled_hours": [str(ts) for ts in history_index[readings.reindex(history_index).isna()]],
            "lightgbm_training_rows": int(len(train)),
            "autoarima_chosen": chosen,
            "scored_hours": {name: scores[name][1] for name in NAMES},
            "hourly": {
                str(ts): {"actual_kwh": helpers.rounded(actual[ts]) if pd.notna(actual[ts]) else None,
                          **{f"{name}_kwh": helpers.rounded(forecasts[name][ts]) for name in NAMES}}
                for ts in target_index
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
