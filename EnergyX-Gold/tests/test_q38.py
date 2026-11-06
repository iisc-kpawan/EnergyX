"""Independent check of Q38: AutoARIMA daily forecasts, London education buildings, 8 to 14 May 2016.

Recomputed with pandas over BDG2's electricity_cleaned.csv and BDG2_metadata.csv, calling
statsforecast through its StatsForecast frame interface: no snapshot, no DuckDB, no
energyx_gold.analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TASK_ID = "Q38"
HISTORY_DAYS = pd.date_range("2016-04-08", periods=30, freq="D")
TARGET_DAYS = pd.date_range("2016-05-08", periods=7, freq="D")


@pytest.fixture(scope="module")
def want(raw) -> dict:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    meta = raw.metadata
    readings = raw.electricity_cleaned
    named = meta.loc[(meta["City"] == "London") & (meta["primaryspaceusage"] == "Education"), "building_id"]
    cohort = sorted(b for b in named if b in readings.columns and readings[b].notna().any())

    hours = readings.loc["2016-04-08":"2016-05-14 23:00", cohort].astype(float)
    counts = hours.notna().resample("D").sum()
    daily = hours.resample("D").sum().where(counts == 24)

    maes = {}
    for building in cohort:
        history = daily[building].reindex(HISTORY_DAYS)
        target = daily[building].reindex(TARGET_DAYS)
        if history.isna().any() or target.isna().any():
            continue
        frame = pd.DataFrame({"unique_id": building, "ds": HISTORY_DAYS, "y": history.to_numpy()})
        forecast = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1).forecast(df=frame, h=7)
        maes[building] = float(np.abs(forecast["AutoARIMA"].to_numpy() - target.to_numpy()).mean())
    return {"cohort": cohort, "maes": maes, "median": float(np.median(list(maes.values())))}


def test_median_matches(gold, want):
    answer = gold(TASK_ID)["result"]["answer"]
    assert answer["median_mae"] == pytest.approx(want["median"], rel=1e-6, abs=5e-7)
    assert answer["buildings"] == len(want["maes"])


def test_every_building_matches(gold, want):
    evidence = gold(TASK_ID)["evidence"]
    assert evidence["cohort"] == want["cohort"]
    assert sorted(evidence["per_building"]) == sorted(want["maes"])
    for building, record in evidence["per_building"].items():
        assert record["mae"] == pytest.approx(want["maes"][building], rel=1e-6, abs=5e-7)
