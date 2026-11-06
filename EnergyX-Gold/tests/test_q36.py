"""Independent check of Q36: seasonal naive, Lamb_education_Norris, 8 May 2016 00:00 to 11:00.

Recomputed with pandas over BDG2's electricity_cleaned.csv: no snapshot, no DuckDB, no
energyx_gold.analytics.
"""

from __future__ import annotations

import pandas as pd
import pytest

TASK_ID = "Q36"
BUILDING = "Lamb_education_Norris"
START = pd.Timestamp("2016-05-08 00:00")
HOURS = 12
SEASON = pd.Timedelta(hours=168)


@pytest.fixture(scope="module")
def want(raw) -> dict:
    values = raw.electricity_cleaned[BUILDING].astype(float)
    target = pd.date_range(START, periods=HOURS, freq="h")
    actual = values.reindex(target)
    forecast = values.reindex(target - SEASON).set_axis(target)
    ok = actual.notna() & forecast.notna()
    return {"mae": float((forecast[ok] - actual[ok]).abs().mean()), "hours": int(ok.sum()),
            "forecast": forecast, "actual": actual}


def test_mae_matches(gold, want):
    result = gold(TASK_ID)["result"]
    assert result["unit"] == "kWh"
    assert result["answer"] == pytest.approx(want["mae"], rel=1e-6, abs=5e-7)


def test_hourly_matches(gold, want):
    evidence = gold(TASK_ID)["evidence"]
    assert evidence["scored_hours"] == want["hours"] == 12
    for ts, row in evidence["hourly"].items():
        assert row["forecast_kwh"] == pytest.approx(want["forecast"][pd.Timestamp(ts)], abs=5e-7)
        assert row["actual_kwh"] == pytest.approx(want["actual"][pd.Timestamp(ts)], abs=5e-7)
