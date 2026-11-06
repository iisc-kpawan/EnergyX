"""Q36: Seasonal naive forecast of the next 12 hours for Lamb_education_Norris from 8 May 2016.

Question: Forecast the next 12 hours of electricity consumption for `Lamb_education_Norris`,
starting at 00:00 on 8 May 2016. Use Seasonal Naive with the previous 7 days of hourly data, and
report the MAE.

Pinned definition: seasonal naive with a 168-hour season. Each hour from 8 May 2016 00:00 to 11:00
is forecast as the reading 168 hours earlier (1 May 2016, same hour). The MAE covers the 12 hours
with both a valid reading and a forecast; the forecasts and errors are in the evidence.

- Times are site-local. The forecast uses the real reading 168 hours earlier. No gap filling is
  pinned, and 1 May 00:00 to 11:00 has no missing hour, so none is needed.

    python -m energyx_gold.tasks.category_08_forecasting.Q36
"""

from __future__ import annotations

import pandas as pd

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_08_forecasting import _helpers as helpers

TASK_ID = "Q36"
BUILDING = "Lamb_education_Norris"
TARGET_START = pd.Timestamp("2016-05-08 00:00")
HORIZON_HOURS = 12
SEASON_HOURS = helpers.WEEK
HISTORY_START = TARGET_START - pd.Timedelta(hours=SEASON_HOURS)


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    target_index = helpers.hours_of(TARGET_START, HORIZON_HOURS)
    readings = series.hourly(context, [BUILDING], HISTORY_START,
                             TARGET_START + pd.Timedelta(hours=HORIZON_HOURS))[BUILDING]
    actual = readings.reindex(target_index)
    forecast = readings.reindex(target_index - pd.Timedelta(hours=SEASON_HOURS)).set_axis(target_index)
    value, scored = helpers.mae(actual, forecast)

    hourly = {
        str(ts): {
            "actual_kwh": None if pd.isna(actual[ts]) else helpers.rounded(actual[ts]),
            "forecast_kwh": None if pd.isna(forecast[ts]) else helpers.rounded(forecast[ts]),
            "abs_error_kwh": None if pd.isna(actual[ts]) or pd.isna(forecast[ts])
            else helpers.rounded(abs(forecast[ts] - actual[ts])),
        }
        for ts in target_index
    }
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": helpers.rounded(value), "unit": "kWh"},
        "evidence": {
            "definition": {
                "building": BUILDING,
                "model": "seasonal naive, y(t - 168 h)",
                "season_hours": SEASON_HOURS,
                "target": f"{target_index[0]} to {target_index[-1]}, site-local",
                "forecast_source": f"{target_index[0] - pd.Timedelta(hours=SEASON_HOURS)} to "
                                   f"{target_index[-1] - pd.Timedelta(hours=SEASON_HOURS)}, real readings",
                "gap_filling": "none",
                "scoring": "hours with both a valid reading and a forecast",
            },
            "history_valid_hours": int(readings.loc[HISTORY_START:TARGET_START - pd.Timedelta(hours=1)].notna().sum()),
            "scored_hours": scored,
            "hourly": hourly,
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
