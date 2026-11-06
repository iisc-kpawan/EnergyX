"""Q40: Seasonal naive against Tiny Time Mixers for the Dublin office buildings, 8 January 2016.

Question: Compare Seasonal Naive and Tiny Time Mixer (TTM) for forecasting the next 24 hours of
electricity consumption for all office buildings in Dublin, starting at 00:00 on 8 January 2016. Use
the previous 7 days of hourly data, and report which model has the lower median MAE across the
buildings.

Pinned definition: office buildings in Dublin (Wolf). A building is forecast when its 168 history
hours (1 January 2016 00:00 to 7 January 23:00) are complete after gaps of up to 24 hours are filled
linearly, and it has 24 valid readings on 8 January 2016. Both models forecast the 24 hours of 8
January: seasonal naive repeats the reading 168 hours earlier, and Tiny Time Mixers (granite-tsfm,
ibm-granite/granite-timeseries-ttm-r2) runs zero-shot on the 168 history hours, using the checkpoint
get_model selects for context length 168 and prediction length 24. The answer is each model's median
MAE over the buildings; the checkpoint and every building's errors are in the evidence.

- get_model selects the 90-30-ft-r2.1 branch (pinned to commit 6e5cb8e). It reads 90 hours of
  context, so TTM sees the last 90 of the 168 history hours (4 January 06:00 to 7 January 23:00);
  given all 168 the model itself keeps only the last 90. It outputs 30 hours, cut to 24 by
  prediction_filter_length.
- The context is standardised by its own mean and population standard deviation (docs/ml-protocol.md),
  zero-shot on CPU with the hourly frequency token.
- Seasonal naive reads the filled history.
- Medians are over the unrounded MAEs.

    python -m energyx_gold.tasks.category_08_forecasting.Q40
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyx_gold.analytics import series
from energyx_gold.core.context import GoldContext
from energyx_gold.tasks.category_08_forecasting import _helpers as helpers

TASK_ID = "Q40"
SITE = "Wolf"
BUILDING_TYPE = "Office"
TARGET_START = pd.Timestamp("2016-01-08 00:00")
HISTORY_HOURS = helpers.WEEK
HORIZON_HOURS = helpers.HOURS
SEASON_HOURS = helpers.WEEK
MAX_FILL_HOURS = helpers.MAX_FILL_HOURS
NAMES = {"seasonal_naive": "Seasonal Naive", "ttm": "TTM"}


def main(context: GoldContext | None = None) -> dict:
    context = context or GoldContext()
    buildings = series.cohort(context, site_id=SITE, building_type_raw=BUILDING_TYPE)
    history_index = helpers.hours_of(TARGET_START - pd.Timedelta(hours=HISTORY_HOURS), HISTORY_HOURS)
    target_index = helpers.hours_of(TARGET_START, HORIZON_HOURS)
    readings = series.hourly(context, buildings, history_index[0], target_index[-1] + pd.Timedelta(hours=1))
    filled = series.fill_gaps(readings.reindex(history_index), limit=MAX_FILL_HOURS)

    eligible, not_eligible, per_building = [], {}, {}
    ttm_context_hours = None
    for building in buildings:
        window = helpers.window_evidence(readings[building], filled[building], history_index)
        actual = readings[building].reindex(target_index)
        if window["missing"] or actual.notna().sum() != HORIZON_HOURS:
            not_eligible[building] = {"history_missing_after_fill": window["missing"],
                                      "valid_target_hours": int(actual.notna().sum())}
            continue
        eligible.append(building)
        history = filled[building]
        ttm_values, ttm_context_hours = helpers.ttm_forecast(history, target_index, seed=context.seed)
        forecasts = {
            "seasonal_naive": history.reindex(target_index - pd.Timedelta(hours=SEASON_HOURS)).set_axis(target_index),
            "ttm": ttm_values,
        }
        per_building[building] = {name: helpers.mae(actual, forecasts[name])[0] for name in NAMES}
        per_building[building]["filled_history_hours"] = window["filled"]

    medians = {name: float(np.median([per_building[b][name] for b in eligible])) for name in NAMES}
    answer = {f"{name}_median_mae": helpers.rounded(medians[name]) for name in NAMES}
    answer["buildings"] = len(eligible)
    answer["lower_median_mae_model"] = NAMES[helpers.lower({n: answer[f"{n}_median_mae"] for n in NAMES})]
    units = {f"{name}_median_mae": "kWh" for name in NAMES} | {"buildings": "count"}
    return {
        "task_id": TASK_ID,
        "status": "success",
        "result": {"answer": answer, "units": units},
        "evidence": {
            "definition": {
                "cohort": {"site_id": SITE, "city": "Dublin", "building_type_raw": BUILDING_TYPE,
                           "with_readings": True},
                "history": f"{history_index[0]} to {history_index[-1]} ({HISTORY_HOURS} hours), site-local",
                "target": f"{target_index[0]} to {target_index[-1]}, site-local",
                "eligibility": "complete history after filling gaps of up to 24 hours, and 24 valid target readings",
                "max_fill_hours": MAX_FILL_HOURS,
                "seasonal_naive": {"season_hours": SEASON_HOURS},
                "ttm": {"model": helpers.TTM_MODEL,
                        "get_model_request": {"context_length": helpers.TTM_REQUESTED_CONTEXT,
                                              "prediction_length": HORIZON_HOURS},
                        "checkpoint_branch": helpers.TTM_BRANCH, "revision": helpers.TTM_REVISION,
                        "context_hours_used": ttm_context_hours,
                        "context_note": "the checkpoint reads 90 hours, so the last 90 of the 168 history hours",
                        "frequency_token": helpers.TTM_HOURLY_FREQ_TOKEN,
                        "standardisation": "context mean and population standard deviation",
                        "device": "cpu, one thread", "training": "zero-shot"},
                "aggregate": "median over the eligible buildings, on unrounded values",
            },
            "cohort_size": len(buildings),
            "cohort": buildings,
            "eligible": eligible,
            "not_eligible": not_eligible,
            "per_building": {
                b: {f"{name}_mae": helpers.rounded(r[name]) for name in NAMES}
                | {"filled_history_hours": r["filled_history_hours"]}
                for b, r in per_building.items()
            },
            "queries": list(context.queries_run),
        },
    }


if __name__ == "__main__":
    from energyx_gold.core.runner import run_standalone

    run_standalone(TASK_ID)
