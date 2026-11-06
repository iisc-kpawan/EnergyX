"""Q38: AutoARIMA daily forecasts for the London education buildings, 8 to 14 May 2016.

Question: Forecast the next 7 days of daily electricity consumption for all education buildings in
London, starting on 8 May 2016. Use AutoARIMA with the previous 30 days of daily data, and report the
median MAE across the buildings.

Pinned definition: London education buildings. A day's consumption is the sum of its 24 hourly
readings, and a day counts only when all 24 are valid. A building is forecast when all 30 history
days (8 April to 7 May 2016) and all 7 target days (8 to 14 May 2016) count. AutoARIMA
(statsforecast, season_length 7, default stepwise search) fitted to the 30 daily values forecasts the
7 target days, and the MAE covers those 7 days. The answer is the median over the forecast
buildings; every building's MAE is in the evidence.

- London is the Mouse, Robin and Shrew sites; only Robin has education buildings.
- Days are site-local calendar days. No gap filling: a day with a missing hour doesn't count.
- The median is taken over the unrounded MAEs.

    python -m energyx_gold.tasks.category_08_forecasting.Q38
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_08_forecasting import _helpers as helpers

TASK_ID = "Q38"
CITY = "London"
BUILDING_TYPE = "Education"
HISTORY_START = pd.Timestamp("2016-04-08")
HISTORY_DAYS = 30
TARGET_START = pd.Timestamp("2016-05-08")
TARGET_DAYS = 7
SEASON_LENGTH = 7


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    buildings = series.cohort(context, city=CITY, building_type_raw=BUILDING_TYPE)
    history_days = pd.date_range(HISTORY_START, periods=HISTORY_DAYS, freq="D")
    target_days = pd.date_range(TARGET_START, periods=TARGET_DAYS, freq="D")
    assert history_days[-1] + pd.Timedelta(days=1) == TARGET_START

    readings = series.hourly(context, buildings, HISTORY_START, target_days[-1] + pd.Timedelta(days=1))
    day = readings.index.normalize()
    daily = readings.groupby(day).sum(min_count=1)
    complete = readings.notna().groupby(day).sum() == helpers.HOURS
    daily = daily.where(complete)

    per_building, not_forecast = {}, {}
    for building in buildings:
        history = daily[building].reindex(history_days)
        target = daily[building].reindex(target_days)
        if history.isna().any() or target.isna().any():
            not_forecast[building] = {"history_days_counted": int(history.notna().sum()),
                                      "target_days_counted": int(target.notna().sum())}
            continue
        forecast, chosen = helpers.autoarima(history.to_numpy(), SEASON_LENGTH, TARGET_DAYS)
        value, _ = helpers.mae(target, pd.Series(forecast, index=target_days))
        per_building[building] = {
            "mae": value,
            "model": chosen["model"],
            "order": chosen["order"],
            "seasonal_order": chosen["seasonal_order"],
            "forecast_kwh": [helpers.rounded(v) for v in forecast],
            "actual_kwh": [helpers.rounded(v) for v in target],
        }

    maes = [per_building[b]["mae"] for b in per_building]
    answer = {"median_mae": helpers.rounded(np.median(maes)), "buildings": len(per_building)}
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": {"median_mae": "kWh", "buildings": "count"}},
        "evidence": {
            "definition": {
                "cohort": {"city": CITY, "building_type_raw": BUILDING_TYPE, "with_readings": True},
                "daily_value": "sum of the 24 hourly readings of a site-local day; a day with a missing hour doesn't count",
                "history": f"{history_days[0].date()} to {history_days[-1].date()} ({HISTORY_DAYS} days)",
                "target": f"{target_days[0].date()} to {target_days[-1].date()} ({TARGET_DAYS} days)",
                "autoarima": {"library": "statsforecast", "season_length": SEASON_LENGTH,
                              "other_settings": "defaults (stepwise search)"},
                "aggregate": "median of the unrounded per-building MAEs",
            },
            "cohort_size": len(buildings),
            "cohort": buildings,
            "not_forecast": not_forecast,
            "mean_mae": helpers.rounded(np.mean(maes)),
            "per_building": {b: {**r, "mae": helpers.rounded(r["mae"])} for b, r in per_building.items()},
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
