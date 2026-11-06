"""Q37: Seasonal naive against AutoARIMA for Lamb_education_Norris on 8 May 2016.

Question: Compare Seasonal Naive and AutoARIMA for forecasting the next 24 hours of electricity
consumption for `Lamb_education_Norris`, starting at 00:00 on 8 May 2016. Use the previous 7 days of
hourly data, and report which method has the lower MAE.

Pinned definition: Lamb_education_Norris. History is the 168 hours from 1 May 2016 00:00 to 7 May
23:00, with gaps of up to 24 hours filled linearly. Both methods forecast the 24 hours of 8 May 2016.
Seasonal naive repeats the reading 168 hours earlier. AutoARIMA is statsforecast's AutoARIMA with
season_length 24 and its default stepwise search. The MAE covers the hours of 8 May with a valid
reading; the chosen ARIMA orders are in the evidence.

- Seasonal naive reads the filled history, so 8 May hour h is 1 May hour h.
- AutoARIMA is fitted to the 168 filled history values and forecasts 24 steps.

    python -m energyx_gold.tasks.category_08_forecasting.Q37
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_08_forecasting import _helpers as helpers

TASK_ID = "Q37"
BUILDING = "Lamb_education_Norris"
TARGET_START = pd.Timestamp("2016-05-08 00:00")
HISTORY_HOURS = helpers.WEEK
HORIZON_HOURS = helpers.HOURS
SEASON_NAIVE_HOURS = helpers.WEEK
ARIMA_SEASON_LENGTH = 24
MAX_FILL_HOURS = helpers.MAX_FILL_HOURS
NAMES = {"seasonal_naive": "Seasonal Naive", "autoarima": "AutoARIMA"}


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    history_index = helpers.hours_of(TARGET_START - pd.Timedelta(hours=HISTORY_HOURS), HISTORY_HOURS)
    target_index = helpers.hours_of(TARGET_START, HORIZON_HOURS)
    readings = series.hourly(context, [BUILDING], history_index[0], target_index[-1] + pd.Timedelta(hours=1))[BUILDING]
    filled = series.fill_gaps(readings.reindex(history_index), limit=MAX_FILL_HOURS)
    if filled.isna().any():
        raise ValueError(f"{BUILDING}: history has a gap longer than {MAX_FILL_HOURS} hours")
    actual = readings.reindex(target_index)

    arima_values, chosen = helpers.autoarima(filled.to_numpy(), ARIMA_SEASON_LENGTH, HORIZON_HOURS)
    forecasts = {
        "seasonal_naive": filled.reindex(target_index - pd.Timedelta(hours=SEASON_NAIVE_HOURS)).set_axis(target_index),
        "autoarima": pd.Series(arima_values, index=target_index),
    }
    scores = {name: helpers.mae(actual, forecast) for name, forecast in forecasts.items()}
    maes = {name: scores[name][0] for name in NAMES}

    answer = {f"{name}_mae": helpers.rounded(maes[name]) for name in NAMES}
    answer["lower_mae_method"] = NAMES[helpers.lower({name: answer[f"{name}_mae"] for name in NAMES})]
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
                "seasonal_naive": {"season_hours": SEASON_NAIVE_HOURS},
                "autoarima": {"library": "statsforecast", "season_length": ARIMA_SEASON_LENGTH,
                              "other_settings": "defaults (stepwise search)"},
                "scoring": "hours of 8 May with a valid reading",
                "tie": "the lower MAE is compared on the rounded values; a tie goes to the name first in alphabetical order",
            },
            "history_window": helpers.window_evidence(readings, filled, history_index),
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
